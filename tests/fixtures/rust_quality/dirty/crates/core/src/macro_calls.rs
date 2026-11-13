//! Planted: a deprecated call inside a local macro, invoked at three call sites
//! (lines 17, 21, 25). rustc reports each at the macro body (line 12) with an
//! `expansion` span pointing back at the call site.

#[deprecated]
pub fn old() -> i32 {
    1
}

macro_rules! call_old {
    () => {
        old()
    };
}

pub fn a() -> i32 {
    call_old!()
}

pub fn b() -> i32 {
    call_old!()
}

pub fn c() -> i32 {
    call_old!()
}
