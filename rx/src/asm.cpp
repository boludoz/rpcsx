#include "asm.hpp"
#include <cstdlib>

#if defined(ARCH_X64) && !defined(_MSC_VER)
#include <cpuid.h>
#endif

namespace {
bool detect_tm() {
  // Escape hatch: transactional elision can be disabled at runtime.
  if (std::getenv("RX_NO_TM")) {
    return false;
  }

#if defined(ARCH_X64)
  u32 eax = 0, ebx = 0, ecx = 0, edx = 0;
#ifdef _MSC_VER
  int regs[4]{};
  __cpuid(regs, 0);
  if (regs[0] < 7) {
    return false;
  }
  __cpuidex(regs, 7, 0);
  ebx = static_cast<u32>(regs[1]);
  edx = static_cast<u32>(regs[3]);
#else
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }
#endif
  // RTM_ALWAYS_ABORT (leaf 7 EDX bit 11): TSX fused off by microcode, every
  // xbegin aborts. Treat as unsupported to avoid paying the abort path.
  if (edx & (1u << 11)) {
    return false;
  }

  // RTM: leaf 7 EBX bit 11
  return (ebx & (1u << 11)) != 0;
#elif defined(ARCH_ARM64) && defined(__linux__)
  // FEAT_TME: ID_AA64ISAR0_EL1[27:24] != 0. EL0 reads of the ID registers
  // are trapped and emulated by Linux since 4.11, and the kernel only
  // reports features it supports, so this is safe and conservative.
  u64 isar0 = 0;
  __asm__ volatile("mrs %0, s3_0_c0_c6_0" : "=r"(isar0));
  return ((isar0 >> 24) & 0xf) != 0;
#else
  return false;
#endif
}

#ifdef ARCH_ARM64
u64 detect_arm_tsc_scale() {
  u64 freq = 0;
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));

  // Scale the hardware counter so busy_wait cycle counts written for a ~3GHz
  // x86 TSC keep roughly the same duration (see busy_wait: cycles / 100).
  const u64 scale = freq / 30'000'000;
  return scale ? scale : 1;
}
#endif
} // namespace

bool g_use_rtm = detect_tm();
u64 g_rtm_tx_limit1 = 65535;

#ifdef ARCH_ARM64
u64 g_arm_tsc_scale = detect_arm_tsc_scale();
#endif
