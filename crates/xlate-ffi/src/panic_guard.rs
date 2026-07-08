use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub fn guard_ptr<T>(f: impl FnOnce() -> *mut T) -> *mut T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(_) => {
            tracing::error!("panic caught at FFI boundary; returning null");
            std::ptr::null_mut()
        }
    }
}

pub fn guard_cstr(f: impl FnOnce() -> *mut c_char) -> *mut c_char {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(_) => {
            tracing::error!("panic caught at FFI boundary; returning null");
            std::ptr::null_mut()
        }
    }
}
