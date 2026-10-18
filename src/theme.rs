use crate::config::AppAppearance;
use objc2::rc::Retained;
use objc2_app_kit::{
    NSAppearance, NSAppearanceNameAqua, NSAppearanceNameDarkAqua, NSApplication, NSColor,
};
use objc2_foundation::MainThreadMarker;

pub fn apply_app_appearance(mtm: MainThreadMarker, appearance: AppAppearance) {
    let app = NSApplication::sharedApplication(mtm);
    if let Some(ns_appearance) = ns_appearance(appearance) {
        app.setAppearance(Some(&ns_appearance));
    }
}

pub fn title_text(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(0.97, 0.95, 0.92, 1.0),
        AppAppearance::Light => srgb(0.15, 0.11, 0.08, 1.0),
    }
}

pub fn secondary_text(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(0.82, 0.78, 0.73, 1.0),
        AppAppearance::Light => srgb(0.39, 0.31, 0.24, 1.0),
    }
}

pub fn logo_badge_background(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(0.12, 0.10, 0.08, 0.0),
        AppAppearance::Light => srgb(0.06, 0.05, 0.05, 0.98),
    }
}

pub fn logo_badge_border(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(0.0, 0.0, 0.0, 0.0),
        AppAppearance::Light => srgb(0.86, 0.70, 0.34, 0.22),
    }
}

pub fn overlay_transcript_text(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(1.0, 1.0, 1.0, 0.90),
        AppAppearance::Light => srgb(0.08, 0.08, 0.08, 0.92),
    }
}

pub fn overlay_panel_fill(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(0.08, 0.06, 0.05, 0.22),
        AppAppearance::Light => srgb(1.0, 0.99, 0.97, 0.48),
    }
}

pub fn overlay_panel_border(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => srgb(0.86, 0.70, 0.34, 0.12),
        AppAppearance::Light => srgb(0.14, 0.12, 0.10, 0.12),
    }
}

pub fn overlay_bar_color(appearance: AppAppearance, glow: f64) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => {
            let r = 0.98;
            let g = 0.80 + 0.16 * glow;
            let b = 0.22 + 0.08 * glow;
            let alpha = 0.40 + 0.24 * glow;
            NSColor::colorWithRed_green_blue_alpha(r, g, b, alpha)
        }
        AppAppearance::Light => {
            let r = 0.73 + 0.08 * glow;
            let g = 0.52 + 0.10 * glow;
            let b = 0.10 + 0.05 * glow;
            let alpha = 0.66 + 0.18 * glow;
            NSColor::colorWithRed_green_blue_alpha(r, g, b, alpha)
        }
    }
}

pub fn loading_panel_background(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => NSColor::colorWithCalibratedWhite_alpha(0.08, 0.96),
        AppAppearance::Light => srgb(0.99, 0.98, 0.95, 0.98),
    }
}

pub fn loading_divider(appearance: AppAppearance) -> Retained<NSColor> {
    match appearance {
        AppAppearance::Dark => NSColor::colorWithCalibratedWhite_alpha(1.0, 0.12),
        AppAppearance::Light => srgb(0.12, 0.10, 0.08, 0.10),
    }
}

fn ns_appearance(appearance: AppAppearance) -> Option<Retained<NSAppearance>> {
    let name = unsafe {
        match appearance {
            AppAppearance::Dark => NSAppearanceNameDarkAqua,
            AppAppearance::Light => NSAppearanceNameAqua,
        }
    };
    NSAppearance::appearanceNamed(name)
}

fn srgb(r: f64, g: f64, b: f64, a: f64) -> Retained<NSColor> {
    NSColor::colorWithSRGBRed_green_blue_alpha(r, g, b, a)
}
