//! Type-safe kernel message passing between processes.
//!
//! Zero-copy send, auto-free receive. Works in `no_std` contexts.

use crate::Pid;
use crate::syscall;

/// Wire format for the send/recv message syscalls.
///
/// Both kernel and userland read/write this layout through user pointers.
/// For sends, `data`/`len` point to the payload in the sender's address space.
/// For receives, the kernel allocates `data` on the receiver's heap.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RawMessage {
    pub sender: u32,
    pub msg_type: u32,
    pub data: u64,
    pub len: u64,
}

/// A received message. Payload buffer is kernel-allocated and freed on drop.
pub struct ReceivedMessage {
    raw: RawMessage,
}

impl ReceivedMessage {
    /// Raw pointer to the payload buffer (kernel-allocated on receive).
    pub fn data_ptr(&self) -> *const u8 {
        core::ptr::with_exposed_provenance(self.raw.data as usize)
    }

    /// Payload size in bytes.
    pub fn data_len(&self) -> usize {
        self.raw.len as usize
    }

    /// Sender PID.
    pub fn sender(&self) -> u32 {
        self.raw.sender
    }

    /// Message type tag.
    pub fn msg_type(&self) -> u32 {
        self.raw.msg_type
    }

    /// Payload as a byte slice.
    pub fn bytes(&self) -> &[u8] {
        if self.raw.len == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.data_len()) }
    }

    /// Read the typed payload. Buffer is freed when this message drops.
    ///
    /// # Panics
    /// Panics if the payload is smaller than `size_of::<T>()`.
    pub fn payload<T: Copy>(&self) -> T {
        let expected = core::mem::size_of::<T>();
        if expected == 0 {
            return unsafe { core::mem::zeroed() };
        }
        assert!(
            self.raw.len as usize >= expected,
            "message payload too small: got {}, expected {expected}",
            self.raw.len,
        );
        // Use read_unaligned: kernel allocates with align=1.
        unsafe { core::ptr::read_unaligned(self.raw.data as *const T) }
    }
}

impl Drop for ReceivedMessage {
    fn drop(&mut self) {
        if self.raw.data != 0 && self.raw.len != 0 {
            // SAFETY: data/len came from the kernel's recv_msg allocation.
            unsafe {
                syscall::free(
                    core::ptr::with_exposed_provenance_mut(self.raw.data as usize),
                    self.raw.len as usize,
                    1,
                );
            }
        }
    }
}

/// Send a typed payload to another process. The kernel copies directly from
/// `payload` during the syscall — zero allocation, zero copy.
pub fn send<T>(target: Pid, msg_type: u32, payload: &T) {
    let msg = RawMessage {
        sender: 0,
        msg_type,
        data: payload as *const T as u64,
        len: core::mem::size_of::<T>() as u64,
    };
    unsafe { syscall::send_msg(target.0 as u64, &msg as *const RawMessage as u64) };
}

/// Send a variable-length byte payload. The kernel copies from `data` during
/// the syscall — no heap allocation needed.
pub fn send_bytes(target: Pid, msg_type: u32, data: &[u8]) {
    let msg = RawMessage {
        sender: 0,
        msg_type,
        data: data.as_ptr() as u64,
        len: data.len() as u64,
    };
    unsafe { syscall::send_msg(target.0 as u64, &msg as *const RawMessage as u64) };
}

/// Send a message with no payload.
pub fn signal(target: Pid, msg_type: u32) {
    let msg = RawMessage { sender: 0, msg_type, data: 0, len: 0 };
    unsafe { syscall::send_msg(target.0 as u64, &msg as *const RawMessage as u64) };
}

/// Receive the next message (blocks if queue is empty).
pub fn recv() -> ReceivedMessage {
    let mut raw = RawMessage { sender: 0, msg_type: 0, data: 0, len: 0 };
    unsafe { syscall::recv_msg(&mut raw as *mut RawMessage as u64) };
    ReceivedMessage { raw }
}
