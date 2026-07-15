#pragma once

#include "rx/tsc.hpp"
#include "types.hpp"
#include <atomic>
#include <chrono>
#include <thread>

#if defined(ARCH_X64) && !defined(_MSC_VER)
#include <cpuid.h>
// _umonitor/_umwait (WAITPKG) live here, not in <immintrin.h>; mwaitx is
// pulled in by <immintrin.h> already.
#if __has_include(<x86gprintrin.h>)
#include <x86gprintrin.h>
#endif
#endif

// True when hardware transactional memory is usable:
// Intel TSX (RTM) on x86-64, FEAT_TME on ARM64. Defined in asm.cpp.
extern bool g_use_rtm;
// Suggested transaction attempt budget for spin loops (defined in asm.cpp).
extern u64 g_rtm_tx_limit1;

#ifdef ARCH_ARM64
// cntvct_el0 usually ticks at 19-100MHz instead of the ~3GHz a TSC-based
// busy_wait assumes; this scale (cntfrq/30MHz, min 1) compensates.
extern u64 g_arm_tsc_scale;
#endif

#ifdef _M_X64
#ifdef _MSC_VER
extern "C" {
u32 _xbegin();
void _xend();
void _xabort(unsigned int);
unsigned char _xtest(void);
void _mm_pause();
void _mm_prefetch(const char *, int);
void _m_prefetchw(const volatile void *);

uchar _rotl8(uchar, uchar);
ushort _rotl16(ushort, uchar);
u64 __popcnt64(u64);

s64 __mulh(s64, s64);
u64 __umulh(u64, u64);

s64 _div128(s64, s64, s64, s64 *);
u64 _udiv128(u64, u64, u64, u64 *);
void __debugbreak();
}
#include <intrin.h>
#include <immintrin.h>
#else
#include <immintrin.h>
#endif
#endif

#ifdef __ARM_FEATURE_TME
#include <arm_acle.h>
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

namespace rx {
// Try to prefetch to Level 2 cache since it's not split to data/code on most
// processors
template <typename T> constexpr void prefetch_exec(T func) {
  if (std::is_constant_evaluated()) {
    return;
  }

  const u64 value = reinterpret_cast<u64>(func);
  const void *ptr = reinterpret_cast<const void *>(value);

#ifdef _M_X64
  return _mm_prefetch(static_cast<const char *>(ptr), _MM_HINT_T1);
#else
  return __builtin_prefetch(ptr, 0, 2);
#endif
}

// Try to prefetch to Level 1 cache
constexpr void prefetch_read(const void *ptr) {
  if (std::is_constant_evaluated()) {
    return;
  }

#ifdef _M_X64
  return _mm_prefetch(static_cast<const char *>(ptr), _MM_HINT_T0);
#else
  return __builtin_prefetch(ptr, 0, 3);
#endif
}

constexpr void prefetch_write(void *ptr) {
  if (std::is_constant_evaluated()) {
    return;
  }

#if defined(_M_X64) && !defined(__clang__)
  return _m_prefetchw(ptr);
#else
  // Locality 3: keep the line resident in L1; locality 0 hinted an immediate
  // eviction which defeats the purpose of prefetching before a write.
  return __builtin_prefetch(ptr, 1, 3);
#endif
}

constexpr u8 rol8(u8 x, u8 n) {
  if (std::is_constant_evaluated()) {
    return (x << (n & 7)) | (x >> ((-n & 7)));
  }

#ifdef _MSC_VER
  return _rotl8(x, n);
#elif defined(__clang__)
  return __builtin_rotateleft8(x, n);
#elif defined(ARCH_X64)
  return __builtin_ia32_rolqi(x, n);
#else
  return (x << (n & 7)) | (x >> ((-n & 7)));
#endif
}

constexpr u16 rol16(u16 x, u16 n) {
  if (std::is_constant_evaluated()) {
    return (x << (n & 15)) | (x >> ((-n & 15)));
  }

#ifdef _MSC_VER
  return _rotl16(x, static_cast<uchar>(n));
#elif defined(__clang__)
  return __builtin_rotateleft16(x, n);
#elif defined(ARCH_X64)
  return __builtin_ia32_rolhi(x, n);
#else
  return (x << (n & 15)) | (x >> ((-n & 15)));
#endif
}

constexpr u32 rol32(u32 x, u32 n) {
  if (std::is_constant_evaluated()) {
    return (x << (n & 31)) | (x >> (((0 - n) & 31)));
  }

#ifdef _MSC_VER
  return _rotl(x, n);
#elif defined(__clang__)
  return __builtin_rotateleft32(x, n);
#else
  return (x << (n & 31)) | (x >> (((0 - n) & 31)));
#endif
}

constexpr u64 rol64(u64 x, u64 n) {
  if (std::is_constant_evaluated()) {
    return (x << (n & 63)) | (x >> (((0 - n) & 63)));
  }

#ifdef _MSC_VER
  return _rotl64(x, static_cast<int>(n));
#elif defined(__clang__)
  return __builtin_rotateleft64(x, n);
#else
  return (x << (n & 63)) | (x >> (((0 - n) & 63)));
#endif
}

constexpr u32 popcnt64(u64 v) {
#if !defined(_MSC_VER) || defined(__SSE4_2__)
  if (std::is_constant_evaluated())
#endif
  {
    v = (v & 0xaaaaaaaaaaaaaaaa) / 2 + (v & 0x5555555555555555);
    v = (v & 0xcccccccccccccccc) / 4 + (v & 0x3333333333333333);
    v = (v & 0xf0f0f0f0f0f0f0f0) / 16 + (v & 0x0f0f0f0f0f0f0f0f);
    v = (v & 0xff00ff00ff00ff00) / 256 + (v & 0x00ff00ff00ff00ff);
    v = ((v & 0xffff0000ffff0000) >> 16) + (v & 0x0000ffff0000ffff);
    return static_cast<u32>((v >> 32) + v);
  }

#if !defined(_MSC_VER) || defined(__SSE4_2__)
#ifdef _MSC_VER
  return static_cast<u32>(__popcnt64(v));
#else
  return __builtin_popcountll(v);
#endif
#endif
}

constexpr u32 popcnt128(const u128 &v) {
#ifdef _MSC_VER
  return popcnt64(v.lo) + popcnt64(v.hi);
#else
  return popcnt64(v) + popcnt64(v >> 64);
#endif
}

constexpr u64 umulh64(u64 x, u64 y) {
#ifdef _MSC_VER
  if (std::is_constant_evaluated())
#endif
  {
    return static_cast<u64>((u128{x} * u128{y}) >> 64);
  }

#ifdef _MSC_VER
  return __umulh(x, y);
#endif
}

inline s64 mulh64(s64 x, s64 y) {
#ifdef _MSC_VER
  return __mulh(x, y);
#else
  return (s128{x} * s128{y}) >> 64;
#endif
}

inline s64 div128(s64 high, s64 low, s64 divisor, s64 *remainder = nullptr) {
#ifdef _MSC_VER
  s64 rem = 0;
  s64 r = _div128(high, low, divisor, &rem);

  if (remainder) {
    *remainder = rem;
  }
#else
  const s128 x = (u128{static_cast<u64>(high)} << 64) | u64(low);
  const s128 r = x / divisor;

  if (remainder) {
    *remainder = x % divisor;
  }
#endif
  return r;
}

inline u64 udiv128(u64 high, u64 low, u64 divisor, u64 *remainder = nullptr) {
#ifdef _MSC_VER
  u64 rem = 0;
  u64 r = _udiv128(high, low, divisor, &rem);

  if (remainder) {
    *remainder = rem;
  }
#else
  const u128 x = (u128{high} << 64) | low;
  const u128 r = x / divisor;

  if (remainder) {
    *remainder = x % divisor;
  }
#endif
  return r;
}

#ifdef _MSC_VER
inline u128 operator/(u128 lhs, u64 rhs) {
  u64 rem = 0;
  return _udiv128(lhs.hi, lhs.lo, rhs, &rem);
}
#endif

constexpr u32 ctz128(u128 arg) {
#ifdef _MSC_VER
  if (!arg.lo)
    return std::countr_zero(arg.hi) + 64u;
  else
    return std::countr_zero(arg.lo);
#else
  if (u64 lo = static_cast<u64>(arg))
    return std::countr_zero<u64>(lo);
  else
    return std::countr_zero<u64>(arg >> 64) + 64;
#endif
}

constexpr u32 clz128(u128 arg) {
#ifdef _MSC_VER
  if (arg.hi)
    return std::countl_zero(arg.hi);
  else
    return std::countl_zero(arg.lo) + 64;
#else
  if (u64 hi = static_cast<u64>(arg >> 64))
    return std::countl_zero<u64>(hi);
  else
    return std::countl_zero<u64>(arg) + 64;
#endif
}

inline void pause() {
#if defined(ARCH_ARM64)
  // "yield" is a NOP on most ARM cores, so spin loops built on it hammer the
  // memory system and starve SMT siblings. "isb" stalls the pipeline for a
  // duration comparable to x86 "pause".
  __asm__ volatile("isb" ::: "memory");
#elif defined(_M_X64)
  _mm_pause();
#elif defined(ARCH_X64)
  __builtin_ia32_pause();
#else
#error "Missing pause() implementation"
#endif
}

inline void yield() { std::this_thread::yield(); }

// Synchronization helper (cache-friendly busy waiting).
// "cycles" is expressed in ~3GHz TSC cycles as on x86.
inline void busy_wait(usz cycles = 3000) {
#ifdef ARCH_ARM64
  // cntvct_el0 ticks far slower than an x86 TSC (e.g. 19.2MHz on Snapdragon
  // 8 gen 2). Without rescaling, a "nanoseconds" wait becomes microseconds.
  const u64 stop = get_tsc() + (cycles / 100) * g_arm_tsc_scale;
#else
  const u64 stop = get_tsc() + cycles;
#endif
  do
    pause();
  while (get_tsc() < stop);
}

#if defined(ARCH_X64)
// WAITPKG (umonitor/umwait/tpause): leaf 7 subleaf 0, ECX bit 5.
inline bool has_waitpkg() {
  static const bool result = [] {
#ifdef _MSC_VER
    int regs[4]{};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
      return false;
    }
    __cpuidex(regs, 7, 0);
    return (static_cast<u32>(regs[2]) & (1u << 5)) != 0;
#else
    u32 eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
      return false;
    }
    return (ecx & (1u << 5)) != 0;
#endif
  }();
  return result;
}

// MONITORX/MWAITX (AMD): extended leaf 0x80000001, ECX bit 29.
inline bool has_waitx() {
  static const bool result = [] {
#ifdef _MSC_VER
    int regs[4]{};
    __cpuid(regs, 0x80000001);
    return (static_cast<u32>(regs[2]) & (1u << 29)) != 0;
#else
    u32 eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx)) {
      return false;
    }
    return (ecx & (1u << 29)) != 0;
#endif
  }();
  return result;
}

// TSC tick rate in Hz, used to translate a microsecond timeout into the
// cycle-count deadline that umwait/mwaitx expect. 0 means "unknown".
inline u64 get_tsc_freq() {
  static const u64 freq = [] {
#ifndef _MSC_VER
    // CPUID leaf 0x15 (Skylake+): freq = crystal_hz * ebx / eax.
    u32 eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(0x15, 0, &eax, &ebx, &ecx, &edx) && eax && ebx &&
        ecx) {
      return static_cast<u64>(ecx) * ebx / eax;
    }
#endif
    // Fallback: calibrate against the (already-synced) steady_clock over a
    // short busy window. Runs once and is cached in this static.
    const u64 tsc0 = get_tsc();
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 <
           std::chrono::milliseconds(1)) {
    }
    const u64 tsc1 = get_tsc();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    if (ns <= 0) {
      return u64{0};
    }
    return static_cast<u64>((tsc1 - tsc0) * 1'000'000'000ull /
                            static_cast<u64>(ns));
  }();
  return freq;
}

inline u64 get_wait_cycles(u64 timeout_us, u64 tsc_freq) {
  constexpr u64 max_timeout = u64{umax};

  if (!tsc_freq) {
    return 0;
  }

  if (timeout_us == max_timeout) {
    return max_timeout;
  }

  const u64 seconds = timeout_us / 1'000'000;
  const u64 micros = timeout_us % 1'000'000;

  if (seconds > max_timeout / tsc_freq) {
    return max_timeout;
  }

  const u64 sec_cycles = seconds * tsc_freq;
  const u64 cycles_per_us = tsc_freq / 1'000'000;

  if (micros && cycles_per_us > max_timeout / micros) {
    return max_timeout;
  }

  const u64 us_cycles = micros * cycles_per_us +
                        (micros * (tsc_freq % 1'000'000)) / 1'000'000;
  return sec_cycles > max_timeout - us_cycles ? max_timeout
                                              : sec_cycles + us_cycles;
}
#endif // ARCH_X64

// True when monitor_wait32 below actually parks on the cacheline (WFE on
// ARM, umwait/mwaitx on x86). When false it degrades to a plain yield, and
// callers with a hot bounded retry loop may prefer busy_wait instead.
inline bool has_monitor_wait() {
#if defined(ARCH_ARM64)
  return true;
#elif defined(ARCH_X64)
  return has_waitpkg() || has_waitx();
#else
  return false;
#endif
}

// Arms a monitor on the word at `addr` and sleeps until something writes to
// that cacheline, the timeout elapses, or (lacking WAITPKG/MWAITX/on other
// architectures) falls back to a plain thread yield. This lets the core idle
// instead of polling, which is both faster to react to a wakeup than a
// futex/WaitOnAddress syscall and much cheaper on power/SMT-sibling
// throughput than a pure pause() spin.
//
// Never a false negative, but can return spuriously with `old_value` still
// current -- callers must always re-check afterward.
inline void monitor_wait32(const std::atomic<u32> &var, u32 old_value,
                           u64 timeout_us) {
#if defined(ARCH_ARM64)
  (void)timeout_us; // WFE also wakes periodically via the event stream

  const auto *wait_addr = reinterpret_cast<const volatile u32 *>(&var);
  u32 value;
  __asm__ volatile("ldaxr %w0, %1" : "=r"(value) : "Q"(*wait_addr) : "memory");

  if (value != old_value) {
    __asm__ volatile("clrex" ::: "memory");
    return;
  }

  __asm__ volatile("wfe" ::: "memory");
  __asm__ volatile("clrex" ::: "memory");
#elif defined(ARCH_X64)
  static const bool use_umwait = has_waitpkg();
  static const bool use_waitx = has_waitx();

  const void *addr = &var;
  const u64 cycles = get_wait_cycles(timeout_us, get_tsc_freq());

  if (use_umwait && cycles) {
#ifdef _MSC_VER
    _umonitor(const_cast<void *>(addr));
#else
    // umonitor %rax (0xf3, 0x0f, 0xae, 0xf0)
    __asm__ volatile(".byte 0xf3, 0x0f, 0xae, 0xf0" ::"a"(addr) : "memory");
#endif

    if (var.load(std::memory_order::relaxed) != old_value) {
      return;
    }

    constexpr u64 max_timeout = u64{umax};
    const u64 now = get_tsc();
    const u64 deadline = cycles > max_timeout - now ? max_timeout : now + cycles;

#ifdef _MSC_VER
    _umwait(0, deadline);
#else
    // umwait %ecx (0xf2, 0x0f, 0xae, 0xf1)
    __asm__ volatile(".byte 0xf2, 0x0f, 0xae, 0xf1"
                     :: "c"(0u), 
                        "a"(static_cast<u32>(deadline)), 
                        "d"(static_cast<u32>(deadline >> 32))
                     : "memory");
#endif
  } else if (use_waitx && cycles) {
#ifdef _MSC_VER
    _mm_monitorx(const_cast<void *>(addr), 0, 0);
#else
    // monitorx (implicit %rax, %rcx, %rdx)
    __asm__ volatile(".byte 0x0f, 0x01, 0xfa" 
                     :: "a"(addr), "c"(0u), "d"(0u) 
                     : "memory");
#endif

    if (var.load(std::memory_order::relaxed) != old_value) {
      return;
    }

    constexpr u32 timer_enable = 2;
    u32 cycles_u32 = cycles > u32{umax} ? u32{umax} : static_cast<u32>(cycles);

#ifdef _MSC_VER
    _mm_mwaitx(timer_enable, 0, cycles_u32);
#else
    // mwaitx (implicit %rax, %rcx, %rbx)
    __asm__ volatile(".byte 0x0f, 0x01, 0xfb" 
                     :: "a"(0u), "c"(timer_enable), "b"(cycles_u32) 
                     : "memory");
#endif
  } else {
    yield();
  }
#else
  (void)var;
  (void)old_value;
  (void)timeout_us;
  yield();
#endif
}

// --- Hardware transactional memory (Intel TSX RTM on x86, FEAT_TME on ARM) --
//
// Unified interface; only call these when g_use_rtm is true, otherwise the
// instructions are undefined on the current CPU.
//
// tx_start() returns tx_started (0) when speculative execution begins. On
// abort, execution resumes at the tx_start call site with all transactional
// effects rolled back and a non-zero status describing the cause.

constexpr u32 tx_started = 0;
constexpr u32 tx_abort_retry = 1u << 0;    // transient abort, retry may help
constexpr u32 tx_abort_explicit = 1u << 1; // tx_cancel() was executed
constexpr u32 tx_abort_other = 1u << 31;   // ensures abort status is non-zero

#if defined(ARCH_X64)
inline u32 tx_start() {
#ifdef _MSC_VER
  const u32 code = _xbegin();
#else
  u32 code = 0xffffffffu; // _XBEGIN_STARTED
  // xbegin +0 encoded as raw bytes so no -mrtm/target attribute is required
  __asm__ volatile(".byte 0xc7,0xf8,0x00,0x00,0x00,0x00" : "+a"(code)::"memory");
#endif
  if (code == 0xffffffffu) {
    return tx_started;
  }

  u32 status = tx_abort_other;
  if (code & (1u << 1)) // _XABORT_RETRY
    status |= tx_abort_retry;
  if (code & (1u << 0)) // _XABORT_EXPLICIT
    status |= tx_abort_explicit;
  return status;
}

inline void tx_commit() {
#ifdef _MSC_VER
  _xend();
#else
  __asm__ volatile(".byte 0x0f,0x01,0xd5" ::: "memory"); // xend
#endif
}

inline void tx_cancel() {
#ifdef _MSC_VER
  _xabort(0xff);
#else
  __asm__ volatile(".byte 0xc6,0xf8,0xff" ::: "memory"); // xabort $0xff
#endif
}

inline bool tx_active() {
#ifdef _MSC_VER
  return _xtest() != 0;
#else
  unsigned char in_tx;
  // xtest clears ZF when executed inside a transaction
  __asm__ volatile(".byte 0x0f,0x01,0xd6\n\tsetnz %0"
                   : "=r"(in_tx)::"cc", "memory");
  return in_tx != 0;
#endif
}
#elif defined(ARCH_ARM64)
inline u32 tx_start() {
  u64 code;
#ifdef __ARM_FEATURE_TME
  code = __tstart();
#else
  // tstart x0 (raw encoding so no -march=...+tme is required)
  register u64 x0_out __asm__("x0");
  __asm__ volatile(".inst 0xd5233060" : "=r"(x0_out)::"memory");
  code = x0_out;
#endif
  if (code == 0) {
    return tx_started;
  }

  u32 status = tx_abort_other;
  if (code & (1u << 15)) // _TMFAILURE_RTRY
    status |= tx_abort_retry;
  if (code & (1u << 16)) // _TMFAILURE_CNCL
    status |= tx_abort_explicit;
  return status;
}

inline void tx_commit() {
#ifdef __ARM_FEATURE_TME
  __tcommit();
#else
  __asm__ volatile(".inst 0xd503307f" ::: "memory"); // tcommit
#endif
}

inline void tx_cancel() {
#ifdef __ARM_FEATURE_TME
  __tcancel(0);
#else
  __asm__ volatile(".inst 0xd4600000" ::: "memory"); // tcancel #0
#endif
}

inline bool tx_active() {
#ifdef __ARM_FEATURE_TME
  return __ttest() != 0;
#else
  // ttest x0: returns the transactional nesting depth (0 = not in txn)
  register u64 x0_out __asm__("x0");
  __asm__ volatile(".inst 0xd5233160" : "=r"(x0_out)::"memory");
  return x0_out != 0;
#endif
}
#else
inline u32 tx_start() { return tx_abort_other; }
inline void tx_commit() {}
inline void tx_cancel() {}
inline bool tx_active() { return false; }
#endif

// Align to power of 2
template <typename T, typename U>
  requires std::is_unsigned_v<T>
constexpr std::make_unsigned_t<std::common_type_t<T, U>> align(T value,
                                                               U align) {
  return static_cast<std::make_unsigned_t<std::common_type_t<T, U>>>(
      (value + (align - 1)) & (T{0} - align));
}

// General purpose aligned division, the result is rounded up not truncated
template <typename T>
  requires std::is_unsigned_v<T>
constexpr T aligned_div(T value, std::type_identity_t<T> align) {
  return static_cast<T>(value / align + T{!!(value % align)});
}

// General purpose aligned division, the result is rounded to nearest
template <typename T>
  requires std::is_integral_v<T>
constexpr T rounded_div(T value, std::type_identity_t<T> align) {
  if constexpr (std::is_unsigned_v<T>) {
    return static_cast<T>(value / align + T{(value % align) > (align / 2)});
  }

  return static_cast<T>(value / align +
                        (value > 0 ? T{(value % align) > (align / 2)}
                                   : 0 - T{(value % align) < (align / 2)}));
}

// Multiplying by ratio, semi-resistant to overflows
template <UnsignedInt T>
constexpr T rational_mul(T value, std::type_identity_t<T> numerator,
                         std::type_identity_t<T> denominator) {
  if constexpr (sizeof(T) <= sizeof(u64) / 2) {
    return static_cast<T>(value * u64{numerator} / u64{denominator});
  }

  return static_cast<T>(value / denominator * numerator +
                        (value % denominator) * numerator / denominator);
}

template <UnsignedInt T> constexpr T add_saturate(T addend1, T addend2) {
  return static_cast<T>(~addend1) < addend2 ? T{umax}
                                            : static_cast<T>(addend1 + addend2);
}

template <UnsignedInt T> constexpr T sub_saturate(T minuend, T subtrahend) {
  return minuend < subtrahend ? T{0} : static_cast<T>(minuend - subtrahend);
}

template <UnsignedInt T> constexpr T mul_saturate(T factor1, T factor2) {
  return factor1 > 0 && T{umax} / factor1 < factor2
             ? T{umax}
             : static_cast<T>(factor1 * factor2);
}

inline void trigger_write_page_fault(void *ptr) {
#if defined(ARCH_X64) && !defined(_MSC_VER)
  __asm__ volatile("lock orl $0, 0(%0)" ::"r"(ptr));
#elif defined(ARCH_ARM64) && !defined(ANDROID)
  u32 value = 0;
  u32 *u32_ptr = static_cast<u32 *>(ptr);
  __asm__ volatile("ldset %w0, %w0, %1"
                   : "+r"(value), "=Q"(*u32_ptr)
                   : "r"(value));
#else
  static_cast<std::atomic<std::uint32_t> *>(ptr)->fetch_or(
      0, std::memory_order::relaxed);
#endif
}
} // namespace rx

#ifdef _MSC_VER
using rx::operator/;
#endif
