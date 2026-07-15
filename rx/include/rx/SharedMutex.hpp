#pragma once

#include "AtomicOp.hpp"
#include "SharedAtomic.hpp"
#include <system_error>

namespace rx {
namespace detail {
// Adaptive lock-elision hints (same idea as glibc's pthread elision): if
// transactions on a given mutex keep failing -- e.g. its critical sections
// are too large, do syscalls, or the lock is genuinely write-contended --
// skip the transactional path for its next few acquisitions instead of
// paying an abort + rollback on every single one.
//
// The counters live in this external table hashed by mutex address, because
// shared_mutex itself must stay a single 32-bit word (its layout is part of
// the IPC/shared-memory contract). Collisions and races only perturb a
// heuristic, so relaxed atomics are sufficient and safe.
inline constexpr unsigned kElisionSkipCount = 3; // acquisitions to skip
inline std::atomic<unsigned char> g_elision_skip[256]{};

inline std::atomic<unsigned char> &elision_hint(const void *ptr) {
  const auto v = reinterpret_cast<uptr>(ptr);
  // >> 4: shared_mutex instances are word-sized and often adjacent; mixing
  // in higher bits spreads neighbors across the table.
  return g_elision_skip[((v >> 4) ^ (v >> 12)) & 255];
}
} // namespace detail

// IPC-ready shared mutex, using only writer lock is recommended
class shared_mutex final {
  friend class shared_cv;

  enum : unsigned {
    c_one = 1u << 14, // Fixed-point 1.0 value (one writer)
    c_sig = 1u << 30,
    c_err = 1u << 31,
  };

  shared_atomic32 m_value{};

  void impl_lock_shared(unsigned val);
  void impl_unlock_shared(unsigned old);
  std::errc impl_wait();
  void impl_signal();
  void impl_lock(unsigned val);
  void impl_unlock(unsigned old);
  void impl_lock_upgrade();

public:
  constexpr shared_mutex() = default;
  shared_mutex(const shared_mutex &) = delete;

  bool try_lock_shared() {
    // Conditional increment
    unsigned value = m_value.load();
    return value < c_one - 1 &&
           m_value.compare_exchange_strong(value, value + 1);
  }

  // Lock with transactional elision (Intel TSX RTM / Arm TME) when available
  void lock_shared() {
    if (g_use_rtm) {
      auto &skip = detail::elision_hint(this);
      if (skip.load(std::memory_order::relaxed) == 0) {
        for (int i = 0; i < 3; i++) {
          const u32 status = tx_start();
          if (status == tx_started) {
            if (m_value.load(std::memory_order::relaxed) < c_one) {
              // Elided: m_value is only read, so concurrent elided readers
              // do not bounce the cacheline. Any real writer modifying
              // m_value aborts this transaction and we re-execute with a
              // real lock.
              return;
            }

            tx_cancel(); // resumes at tx_start above with an abort status
          }

          if (!(status & tx_abort_retry)) {
            break;
          }
        }

        // Elision failed: back off from the transactional path for a while
        skip.store(detail::kElisionSkipCount, std::memory_order::relaxed);
      } else {
        // Load+store instead of fetch_sub: racing decrements may be lost,
        // which is fine for a hint, but can never wrap below zero (which
        // would disable elision for ~256 acquisitions).
        const auto cur = skip.load(std::memory_order::relaxed);
        if (cur > 0) {
          skip.store(cur - 1, std::memory_order::relaxed);
        }
      }
    }

    unsigned value = m_value.load();
    if (value < c_one - 1) [[likely]] {
      unsigned old = value;
      if (m_value.compare_exchange_strong(old, value + 1)) [[likely]] {
        return;
      }
    }

    impl_lock_shared(value + 1);
  }

  void unlock_shared() {
    if (g_use_rtm && tx_active()) {
      tx_commit();
      return;
    }

    const unsigned value = m_value.fetch_add(-1u);
    if (value >= c_one) [[unlikely]] {
      impl_unlock_shared(value);
    }
  }

  bool try_lock() {
    unsigned value = 0;
    return m_value.compare_exchange_strong(value, c_one);
  }

  // Lock with transactional elision (Intel TSX RTM / Arm TME) when available
  void lock() {
    if (g_use_rtm) {
      auto &skip = detail::elision_hint(this);
      if (skip.load(std::memory_order::relaxed) == 0) {
        for (int i = 0; i < 3; i++) {
          const u32 status = tx_start();
          if (status == tx_started) {
            if (m_value.load(std::memory_order::relaxed) == 0) {
              // Elided: the critical section executes speculatively; stores
              // become visible atomically at tx_commit in unlock().
              return;
            }

            tx_cancel(); // resumes at tx_start above with an abort status
          }

          if (!(status & tx_abort_retry)) {
            break;
          }
        }

        // Elision failed: back off from the transactional path for a while
        skip.store(detail::kElisionSkipCount, std::memory_order::relaxed);
      } else {
        // Load+store instead of fetch_sub: racing decrements may be lost,
        // which is fine for a hint, but can never wrap below zero (which
        // would disable elision for ~256 acquisitions).
        const auto cur = skip.load(std::memory_order::relaxed);
        if (cur > 0) {
          skip.store(cur - 1, std::memory_order::relaxed);
        }
      }
    }

    unsigned value = 0;
    if (!m_value.compare_exchange_strong(value, c_one)) [[unlikely]] {
      impl_lock(value);
    }
  }

  void unlock() {
    if (g_use_rtm && tx_active()) {
      tx_commit();
      return;
    }

    const unsigned value = m_value.fetch_add(0u - c_one);
    if (value != c_one) [[unlikely]] {
      impl_unlock(value);
    }
  }

  bool try_lock_upgrade() {
    unsigned value = m_value.load();

    // Conditional increment, try to convert a single reader into a writer,
    // ignoring other writers
    return (value + c_one - 1) % c_one == 0 &&
           m_value.compare_exchange_strong(value, value + c_one - 1);
  }

  void lock_upgrade() {
    if (g_use_rtm && tx_active()) {
      // Upgrading requires mutating m_value, which an elided reader never
      // incremented. Abort the transaction; control returns to lock_shared,
      // which re-acquires a real lock and re-executes this path.
      tx_cancel();
    }

    if (!try_lock_upgrade()) [[unlikely]] {
      impl_lock_upgrade();
    }
  }

  void lock_downgrade() {
    if (g_use_rtm && tx_active()) {
      // Same reasoning as lock_upgrade: fall back to a real lock, otherwise
      // the subtraction below would underflow the untouched m_value.
      tx_cancel();
    }

    // Convert to reader lock (can result in broken state)
    m_value -= c_one - 1;
  }

  // Check whether can immediately obtain an exclusive (writer) lock
  [[nodiscard]] bool is_free() const { return m_value.load() == 0; }

  // Check whether can immediately obtain a shared (reader) lock
  [[nodiscard]] bool is_lockable() const { return m_value.load() < c_one - 1; }

private:
  // For CV
  bool lock_forced(int count = 1);
};

// Simplified shared (reader) lock implementation.
class reader_lock final {
  shared_mutex &m_mutex;
  bool m_upgraded = false;

public:
  reader_lock(const reader_lock &) = delete;
  reader_lock &operator=(const reader_lock &) = delete;
  explicit reader_lock(shared_mutex &mutex) : m_mutex(mutex) {
    m_mutex.lock_shared();
  }

  // One-way lock upgrade; note that the observed state could have been changed
  void upgrade() {
    if (!m_upgraded) {
      m_mutex.lock_upgrade();
      m_upgraded = true;
    }
  }

  // Try to upgrade; if it succeeds, the observed state has NOT been changed
  bool try_upgrade() {
    return m_upgraded || (m_upgraded = m_mutex.try_lock_upgrade());
  }

  ~reader_lock() { m_upgraded ? m_mutex.unlock() : m_mutex.unlock_shared(); }
};

class writer_lock final {
  shared_mutex &m_mutex;

public:
  writer_lock(const writer_lock &) = delete;
  writer_lock &operator=(const writer_lock &) = delete;
  explicit writer_lock(shared_mutex &mutex) : m_mutex(mutex) { m_mutex.lock(); }
  ~writer_lock() { m_mutex.unlock(); }
};
} // namespace rx
