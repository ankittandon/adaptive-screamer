//! The floating "styled suggestion" card. A non-activating panel that appears
//! after a dictation when the styling model proposes a rewrite, and persists
//! independently of the recording overlay until applied, dismissed, or timed out.

use crate::config::AppAppearance;
use crate::theme;
use objc2::rc::Retained;
use objc2::msg_send;
use objc2_app_kit::{
    NSBackingStoreType, NSColor, NSLineBreakMode, NSPanel, NSScreen, NSTextAlignment, NSTextField,
    NSVisualEffectMaterial, NSVisualEffectView, NSWindowStyleMask,
};
use objc2_core_foundation::{CGFloat, CGPoint, CGRect, CGSize};
use objc2_foundation::{MainThreadMarker, NSString};

const WIDTH: f64 = 460.0;
const HEIGHT: f64 = 132.0;
const PADDING: f64 = 16.0;
const CORNER_RADIUS: f64 = 16.0;
const EDGE_MARGIN: f64 = 16.0;
const BOTTOM_MARGIN: f64 = 150.0;

/// Where the suggestion card anchors on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardAnchor {
    /// Top-right under the menu bar — stays out of the typing area.
    TopRight,
    /// Centered near the bottom of the screen.
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
const TITLE_H: f64 = 18.0;
const HINT_H: f64 = 18.0;
const GAP: f64 = 8.0;

pub struct SuggestionCard {
    panel: Retained<NSPanel>,
    effect_view: Retained<NSVisualEffectView>,
    title_label: Retained<NSTextField>,
    text_label: Retained<NSTextField>,
    hint_label: Retained<NSTextField>,
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

        let label = |mtm: MainThreadMarker, y: f64, h: f64, size: f64, lines: isize| {
            let l = NSTextField::labelWithString(&NSString::from_str(""), mtm);
            l.setFrame(CGRect::new(
                CGPoint::new(PADDING, y),
                CGSize::new(WIDTH - PADDING * 2.0, h),
            ));
            l.setDrawsBackground(false);
            l.setBordered(false);
            l.setBezeled(false);
            l.setEditable(false);
            l.setSelectable(false);
            l.setAlignment(NSTextAlignment(0)); // left
            l.setMaximumNumberOfLines(lines);
            l.setLineBreakMode(NSLineBreakMode::ByWordWrapping);
            l.setFont(Some(&objc2_app_kit::NSFont::systemFontOfSize(size)));
            l
        };

        // Bottom-up coordinates.
        let hint_label = label(mtm, PADDING, HINT_H, 11.5, 1);
        let text_y = PADDING + HINT_H + GAP;
        let text_h = HEIGHT - PADDING * 2.0 - HINT_H - GAP - TITLE_H - GAP;
        let text_label = label(mtm, text_y, text_h, 14.0, 3);
        let title_label = label(mtm, HEIGHT - PADDING - TITLE_H, TITLE_H, 11.0, 1);
        title_label.setStringValue(&NSString::from_str("✦ STYLED SUGGESTION"));

        {
            let content_view = panel.contentView().unwrap();
            content_view.addSubview(&effect_view);
            effect_view.addSubview(&title_label);
            effect_view.addSubview(&text_label);
            effect_view.addSubview(&hint_label);
        }

        let card = Self {
            panel,
            effect_view,
            title_label,
            text_label,
            hint_label,
            visible: false,
            appearance,
            anchor,
        };
        card.apply_theme();
        card
    }

    pub fn show(&mut self, styled: &str, hint: &str) {
        self.text_label.setStringValue(&NSString::from_str(styled));
        self.hint_label.setStringValue(&NSString::from_str(hint));
        self.position();
        self.visible = true;
        self.panel.orderFrontRegardless();
        self.panel.setAlphaValue(1.0);
    }

    pub fn hide(&mut self) {
        self.visible = false;
        self.panel.setAlphaValue(0.0);
        self.panel.orderOut(None);
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_appearance(&mut self, appearance: AppAppearance) {
        self.appearance = appearance;
        self.apply_theme();
    }

    fn position(&self) {
        if let Some(mtm) = MainThreadMarker::new() {
            if let Some(screen) = NSScreen::mainScreen(mtm) {
                // visibleFrame excludes the menu bar and dock, so the card stays
                // clear of them.
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
        self.text_label
            .setTextColor(Some(&theme::overlay_transcript_text(self.appearance)));
        self.hint_label
            .setTextColor(Some(&theme::secondary_text(self.appearance)));

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
