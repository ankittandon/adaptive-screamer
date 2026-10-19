//! The floating "styled suggestion" card: a non-activating panel that appears
//! after a dictation when the styling model proposes a rewrite, and persists
//! until applied, dismissed, or timed out.
//!
//! It is interactive — Apply / Ignore buttons post an action the main-thread
//! timer picks up — and also honors the keyboard ⌘V path (detected via AX).

use crate::config::AppAppearance;
use crate::theme;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::AllocAnyThread;
use objc2::runtime::{AnyClass, AnyObject, ClassBuilder, Sel};
use objc2::sel;
use objc2_app_kit::{
    NSAnimationContext, NSBackingStoreType, NSButton, NSColor, NSForegroundColorAttributeName,
    NSLineBreakMode, NSPanel, NSScreen, NSTextAlignment, NSTextField, NSView,
    NSVisualEffectMaterial, NSVisualEffectView, NSWindowStyleMask,
};
use objc2_core_foundation::{CGFloat, CGPoint, CGRect, CGSize};
use objc2_foundation::{MainThreadMarker, NSMutableAttributedString, NSRange, NSString};
use std::cell::Cell;
use std::sync::OnceLock;

const WIDTH: f64 = 430.0;
const HEIGHT: f64 = 150.0;
const CL: f64 = 20.0; // content left (clears the rail)
const CR: f64 = 14.0; // content right
const CORNER_RADIUS: f64 = 16.0;
const EDGE_MARGIN: f64 = 16.0;
const BOTTOM_MARGIN: f64 = 150.0;

/// What the user clicked on the card. The timer polls [`take_card_action`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardAction {
    Apply,
    Dismiss,
}

thread_local! {
    static CARD_ACTION: Cell<Option<CardAction>> = const { Cell::new(None) };
}

/// Consume a pending card action (set by a button click), if any.
pub fn take_card_action() -> Option<CardAction> {
    CARD_ACTION.with(|c| c.take())
}

extern "C" fn apply_card_action(_this: *mut AnyObject, _sel: Sel, _sender: *mut AnyObject) {
    CARD_ACTION.with(|c| c.set(Some(CardAction::Apply)));
}
extern "C" fn dismiss_card_action(_this: *mut AnyObject, _sel: Sel, _sender: *mut AnyObject) {
    CARD_ACTION.with(|c| c.set(Some(CardAction::Dismiss)));
}

fn card_handler() -> *const AnyObject {
    static HANDLER: OnceLock<usize> = OnceLock::new();
    *HANDLER.get_or_init(|| {
        let superclass = AnyClass::get(c"NSObject").unwrap();
        let mut builder = ClassBuilder::new(c"ScreamerCardHandler", superclass).unwrap();
        unsafe {
            builder.add_method(
                sel!(applyCard:),
                apply_card_action as extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
            );
            builder.add_method(
                sel!(dismissCard:),
                dismiss_card_action as extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
            );
        }
        let cls = builder.register();
        let obj: *mut AnyObject = unsafe { msg_send![cls, new] };
        obj as usize
    }) as *const AnyObject
}

/// Where the suggestion card anchors on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardAnchor {
    TopRight,
    BottomCenter,
}

impl CardAnchor {
    pub fn from_config(value: &str) -> Self {
        match value {
            "bottom_center" => CardAnchor::BottomCenter,
            _ => CardAnchor::TopRight,
        }
    }
}

/// Split into alternating runs of word and whitespace, so concatenating the
/// tokens reproduces the input exactly.
fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_ws: Option<bool> = None;
    for ch in s.chars() {
        let ws = ch.is_whitespace();
        if cur_ws == Some(ws) {
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur.push(ch);
            cur_ws = Some(ws);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The styled text, token by token, flagged `true` where the token is new
/// relative to `raw` (a word-level LCS diff). Concatenating the tokens gives
/// back `styled`.
fn styled_segments(raw: &str, styled: &str) -> Vec<(String, bool)> {
    let a = tokenize(raw);
    let b = tokenize(styled);
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut segs = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            segs.push((b[j].clone(), false));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            segs.push((b[j].clone(), true));
            j += 1;
        }
    }
    while j < m {
        segs.push((b[j].clone(), true));
        j += 1;
    }
    segs
}

pub struct SuggestionCard {
    panel: Retained<NSPanel>,
    effect_view: Retained<NSVisualEffectView>,
    rail: Retained<NSView>,
    title_label: Retained<NSTextField>,
    hint_label: Retained<NSTextField>,
    text_label: Retained<NSTextField>,
    apply_btn: Retained<NSButton>,
    ignore_btn: Retained<NSButton>,
    visible: bool,
    appearance: AppAppearance,
    anchor: CardAnchor,
}

impl SuggestionCard {
    pub fn new(mtm: MainThreadMarker, appearance: AppAppearance, anchor: CardAnchor) -> Self {
        let style = NSWindowStyleMask::Borderless | NSWindowStyleMask::NonactivatingPanel;
        let frame = CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(WIDTH, HEIGHT));

        let panel = {
            let panel = NSPanel::initWithContentRect_styleMask_backing_defer(
                mtm.alloc::<NSPanel>(),
                frame,
                style,
                NSBackingStoreType::Buffered,
                false,
            );
            panel.setLevel(25);
            panel.setOpaque(false);
            panel.setBackgroundColor(Some(&NSColor::clearColor()));
            panel.setHasShadow(true);
            panel.setMovableByWindowBackground(false);
            panel.setHidesOnDeactivate(false);
            panel.setFloatingPanel(true);
            // Become key only when a control needs it, so clicking a button does
            // not steal focus from the app the user is typing in.
            panel.setBecomesKeyOnlyIfNeeded(true);
            panel.setAlphaValue(0.0);
            panel.setCollectionBehavior(
                objc2_app_kit::NSWindowCollectionBehavior::CanJoinAllSpaces
                    | objc2_app_kit::NSWindowCollectionBehavior::Stationary
                    | objc2_app_kit::NSWindowCollectionBehavior::IgnoresCycle,
            );
            panel
        };

        let effect_view = {
            let view = NSVisualEffectView::new(mtm);
            view.setFrame(frame);
            view.setMaterial(NSVisualEffectMaterial::HUDWindow);
            view.setBlendingMode(objc2_app_kit::NSVisualEffectBlendingMode::BehindWindow);
            view.setState(objc2_app_kit::NSVisualEffectState::Active);
            view.setWantsLayer(true);
            if let Some(layer) = view.layer() {
                layer.setCornerRadius(CORNER_RADIUS as CGFloat);
                layer.setMasksToBounds(true);
            }
            view
        };

        // Left accent rail.
        let rail = {
            let v = NSView::new(mtm);
            v.setFrame(CGRect::new(
                CGPoint::new(10.0, 12.0),
                CGSize::new(3.0, HEIGHT - 24.0),
            ));
            v.setWantsLayer(true);
            if let Some(layer) = v.layer() {
                layer.setCornerRadius(1.5);
            }
            v
        };

        let make_label = |mtm: MainThreadMarker, x: f64, y: f64, w: f64, h: f64, size: f64, lines: isize, align: i32| {
            let l = NSTextField::labelWithString(&NSString::from_str(""), mtm);
            l.setFrame(CGRect::new(CGPoint::new(x, y), CGSize::new(w, h)));
            l.setDrawsBackground(false);
            l.setBordered(false);
            l.setBezeled(false);
            l.setEditable(false);
            l.setSelectable(false);
            l.setAlignment(NSTextAlignment(align as isize));
            l.setMaximumNumberOfLines(lines);
            l.setLineBreakMode(NSLineBreakMode::ByWordWrapping);
            l.setFont(Some(&objc2_app_kit::NSFont::systemFontOfSize(size)));
            l
        };

        let title_label = make_label(mtm, CL, HEIGHT - 14.0 - 16.0, 150.0, 16.0, 11.0, 1, 0);
        title_label.setStringValue(&NSString::from_str("✦ STYLED"));
        let hint_label = make_label(
            mtm,
            CL + 150.0,
            HEIGHT - 14.0 - 16.0,
            WIDTH - CR - (CL + 150.0),
            16.0,
            10.5,
            1,
            2, // right
        );

        let text_label = make_label(mtm, CL, 54.0, WIDTH - CL - CR, 62.0, 14.0, 3, 0);

        let handler = card_handler();
        let make_btn = |mtm: MainThreadMarker, title: &str, sel: Sel, x: f64, w: f64| {
            let b = unsafe {
                NSButton::buttonWithTitle_target_action(
                    &NSString::from_str(title),
                    Some(&*handler),
                    Some(sel),
                    mtm,
                )
            };
            b.setFrame(CGRect::new(CGPoint::new(x, 12.0), CGSize::new(w, 28.0)));
            b.setBezelStyle(objc2_app_kit::NSBezelStyle::Rounded);
            b.setFont(Some(&objc2_app_kit::NSFont::systemFontOfSize(12.5)));
            b
        };
        let apply_w = 104.0;
        let ignore_w = 80.0;
        let apply_x = WIDTH - CR - apply_w;
        let ignore_x = apply_x - 8.0 - ignore_w;
        let ignore_btn = make_btn(mtm, "Ignore", sel!(dismissCard:), ignore_x, ignore_w);
        let apply_btn = make_btn(mtm, "Apply  ⌘V", sel!(applyCard:), apply_x, apply_w);

        {
            let content_view = panel.contentView().unwrap();
            content_view.addSubview(&effect_view);
            effect_view.addSubview(&rail);
            effect_view.addSubview(&title_label);
            effect_view.addSubview(&hint_label);
            effect_view.addSubview(&text_label);
            effect_view.addSubview(&ignore_btn);
            effect_view.addSubview(&apply_btn);
        }

        let card = Self {
            panel,
            effect_view,
            rail,
            title_label,
            hint_label,
            text_label,
            apply_btn,
            ignore_btn,
            visible: false,
            appearance,
            anchor,
        };
        card.apply_theme();
        card
    }

    pub fn show(&mut self, raw: &str, styled: &str, hint: &str, is_revert: bool) {
        self.set_styled_text(raw, styled);
        self.hint_label.setStringValue(&NSString::from_str(hint));
        let (title, primary, secondary) = if is_revert {
            ("✓ STYLED", "Revert to raw", "Keep")
        } else {
            ("✦ STYLED", "Apply  ⌘V", "Ignore")
        };
        self.title_label.setStringValue(&NSString::from_str(title));
        self.apply_btn.setTitle(&NSString::from_str(primary));
        self.ignore_btn.setTitle(&NSString::from_str(secondary));
        // Clear any stale click action from a previous card.
        let _ = take_card_action();

        self.position();
        self.visible = true;

        // Entrance: fade in and slide gently into place.
        let final_origin = self.panel.frame().origin;
        self.panel.setAlphaValue(0.0);
        self.panel
            .setFrameOrigin(CGPoint::new(final_origin.x, final_origin.y + 10.0));
        self.panel.orderFrontRegardless();
        unsafe {
            NSAnimationContext::beginGrouping();
            let ctx = NSAnimationContext::currentContext();
            ctx.setDuration(0.18);
            let animator: *mut objc2::runtime::AnyObject = msg_send![&*self.panel, animator];
            let _: () = msg_send![animator, setAlphaValue: 1.0f64];
            let _: () = msg_send![animator, setFrameOrigin: final_origin];
            NSAnimationContext::endGrouping();
        }
    }

    /// Set the card text to `styled`, tinting the words that differ from `raw`
    /// so you can see at a glance what the styling changed.
    fn set_styled_text(&self, raw: &str, styled: &str) {
        let ns = NSString::from_str(styled);
        let attr = NSMutableAttributedString::initWithString(NSMutableAttributedString::alloc(), &ns);
        let base = theme::overlay_transcript_text(self.appearance);
        let tint = theme::overlay_bar_color(self.appearance, 1.0);
        let total = styled.encode_utf16().count();
        unsafe {
            let _: () = msg_send![&*attr, addAttribute: NSForegroundColorAttributeName, value: &*base, range: NSRange::new(0, total)];
            let mut off = 0usize;
            for (tok, inserted) in styled_segments(raw, styled) {
                let len = tok.encode_utf16().count();
                if inserted && !tok.trim().is_empty() {
                    let _: () = msg_send![&*attr, addAttribute: NSForegroundColorAttributeName, value: &*tint, range: NSRange::new(off, len)];
                }
                off += len;
            }
        }
        self.text_label.setAttributedStringValue(&attr);
    }

    pub fn hide(&mut self) {
        self.visible = false;
        self.panel.setAlphaValue(0.0);
        self.panel.orderOut(None);
    }

    #[allow(dead_code)]
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    #[allow(dead_code)]
    pub fn set_appearance(&mut self, appearance: AppAppearance) {
        self.appearance = appearance;
        self.apply_theme();
    }

    fn position(&self) {
        if let Some(mtm) = MainThreadMarker::new() {
            if let Some(screen) = NSScreen::mainScreen(mtm) {
                let vf = screen.visibleFrame();
                let (x, y) = match self.anchor {
                    CardAnchor::TopRight => (
                        vf.origin.x + vf.size.width - WIDTH - EDGE_MARGIN,
                        vf.origin.y + vf.size.height - HEIGHT - EDGE_MARGIN,
                    ),
                    CardAnchor::BottomCenter => (
                        vf.origin.x + (vf.size.width - WIDTH) / 2.0,
                        vf.origin.y + BOTTOM_MARGIN,
                    ),
                };
                self.panel.setFrameOrigin(CGPoint::new(x, y));
            }
        }
    }

    fn apply_theme(&self) {
        self.title_label
            .setTextColor(Some(&theme::secondary_text(self.appearance)));
        self.hint_label
            .setTextColor(Some(&theme::secondary_text(self.appearance)));
        self.text_label
            .setTextColor(Some(&theme::overlay_transcript_text(self.appearance)));
        let _ = &self.apply_btn;
        let _ = &self.ignore_btn;

        let accent = theme::overlay_bar_color(self.appearance, 1.0);
        if let Some(layer) = self.rail.layer() {
            unsafe {
                let cg: *const std::ffi::c_void = msg_send![&accent, CGColor];
                let _: () = msg_send![&*layer, setBackgroundColor: cg];
            }
        }

        if let Some(layer) = self.effect_view.layer() {
            layer.setBorderWidth(1.0);
            unsafe {
                let bg = theme::overlay_panel_fill(self.appearance);
                let bg_color: *const std::ffi::c_void = msg_send![&bg, CGColor];
                let border = theme::overlay_panel_border(self.appearance);
                let border_color: *const std::ffi::c_void = msg_send![&border, CGColor];
                let _: () = msg_send![&*layer, setBackgroundColor: bg_color];
                let _: () = msg_send![&*layer, setBorderColor: border_color];
            }
        }
    }
}
