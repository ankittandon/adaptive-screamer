//! Read the text of the frontmost app's focused text field via the macOS
//! Accessibility API.
//!
//! Used by the opt-in telemetry "implicit" capture to grab the final,
//! user-edited message after a dictation without a manual selection or hotkey.
//! Requires Accessibility permission — the same grant the global hotkey and
//! paste already need.

use core_foundation::base::{CFType, CFTypeRef, TCFType};
use core_foundation::string::{CFString, CFStringRef};

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXUIElementCreateSystemWide() -> CFTypeRef;
    fn AXUIElementCopyAttributeValue(
        element: CFTypeRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> i32;
    fn AXUIElementSetMessagingTimeout(element: CFTypeRef, timeout_seconds: f32) -> i32;
}

/// The text content of the frontmost focused text element, if it exposes an
/// `AXValue` string. Returns `None` when nothing is focused, the focused
/// element has no text value, or Accessibility is unavailable.
pub fn focused_text() -> Option<String> {
    unsafe {
        let system_wide_ptr = AXUIElementCreateSystemWide();
        if system_wide_ptr.is_null() {
            return None;
        }
        let system_wide = CFType::wrap_under_create_rule(system_wide_ptr);
        // Bound each query so a busy or unresponsive target app can't hang the
        // polling thread.
        let _ = AXUIElementSetMessagingTimeout(system_wide.as_concrete_TypeRef(), 0.5);

        let focused = copy_attr(system_wide.as_concrete_TypeRef(), "AXFocusedUIElement")?;
        let value = copy_attr(focused.as_concrete_TypeRef(), "AXValue")?;

        if value.instance_of::<CFString>() {
            value.downcast::<CFString>().map(|s| s.to_string())
        } else {
            None
        }
    }
}

unsafe fn copy_attr(element: CFTypeRef, attribute: &str) -> Option<CFType> {
    let attr = CFString::new(attribute);
    let mut out: CFTypeRef = std::ptr::null();
    let err = AXUIElementCopyAttributeValue(element, attr.as_concrete_TypeRef(), &mut out);
    if err != 0 || out.is_null() {
        return None;
    }
    Some(CFType::wrap_under_create_rule(out))
}
