"""Replace calls to the placeholder device function sml_dp4a_us with the PTX
dp4a instruction, which clang 18 has no way to emit.

Semantics were MEASURED on this GPU (dp4a_probe.c), not taken from memory:

    dp4a.u32.s32 %dst, %a, %b, %acc   ->   dst = acc + sum_i u8(a_i) * s8(b_i)

The accumulator is the LAST operand and the first type suffix applies to the
first operand. The placeholder is declared

    unsigned int sml_dp4a_us(unsigned int acc, unsigned int a, unsigned int b)

so a call site maps to  dp4a.u32.s32 %ret, %a, %b, %acc.

    python ptx_dp4a.py IN.ptx OUT.ptx
"""
import re
import sys

src = open(sys.argv[1], encoding="utf-8").read()

decl = re.compile(
    r"\.extern \.func\s*\(\.param \.b32 func_retval0\) sml_dp4a_us\s*\([^()]*\)\s*;",
    re.S)
src, n_decl = decl.subn("", src)

call = re.compile(
    r"\{ // callseq (\d+), 0\s*"
    r"\.reg \.b32 temp_param_reg;\s*"
    r"\.param \.b32 param0;\s*st\.param\.b32\s*\[param0\+0\],\s*([^;]+);\s*"
    r"\.param \.b32 param1;\s*st\.param\.b32\s*\[param1\+0\],\s*([^;]+);\s*"
    r"\.param \.b32 param2;\s*st\.param\.b32\s*\[param2\+0\],\s*([^;]+);\s*"
    r"\.param \.b32 retval0;\s*"
    r"call\.uni \(retval0\),\s*sml_dp4a_us,\s*\(\s*param0,\s*param1,\s*param2\s*\);\s*"
    r"ld\.param\.b32\s*(%\w+),\s*\[retval0\+0\];\s*"
    r"\} // callseq \1",
    re.S)

literal = []


def sub(m):
    acc, a, b, dst = (m.group(2).strip(), m.group(3).strip(),
                      m.group(4).strip(), m.group(5).strip())
    for name, v in (("acc", acc), ("a", a), ("b", b)):
        if not v.startswith("%"):
            literal.append("%s=%s" % (name, v))
    return "dp4a.u32.s32 \t%s, %s, %s, %s;" % (dst, a, b, acc)


src, n_call = call.subn(sub, src)
left = src.count("sml_dp4a_us")
open(sys.argv[2], "w", encoding="utf-8").write(src)

print("  %d extern declaration(s) removed, %d call(s) replaced with dp4a, "
      "%d placeholder reference(s) left" % (n_decl, n_call, left))
if literal:
    print("  %d operand(s) are literals, not registers: %s"
          % (len(literal), ", ".join(sorted(set(literal)))))
if left:
    sys.exit("  placeholder references remain -- the PTX would not load")
