---
title: Deep Analysis Part 2 - The TorchAO Fake-Quantizer Defect
status: draft
created: 2026-11-13
modified: 2026-11-13
---

# 7. The TorchAO asymmetric fake-quantizer defect: real, still live, worth reporting

Part of the [deep analysis, part 2](README.md). It covers your topic 7.
Evidence labels and flags are defined in the [README](README.md#labels-and-flags).

## Short answer

- **It is a real defect.** I reproduced it independently in this session, with
  my own script and my own hand-written reference, on the latest release
  (0.18.0) and on the upstream main branch as of 2026-11-12 (commit
  `4ab9d61`).
- **It is still unfixed and, as far as I could find, unreported.** The faulty
  lines are unchanged on main. Five searches of the upstream issue tracker
  found nothing that matches.
- **An issue is clearly worth filing. A pull request is worth offering, with
  two caveats:** an open upstream pull request edits the same line, and the
  maintainers may prefer to reject the configuration rather than fix it.
- **The draft from 2026-11-01 is accurate but should be updated** with three
  things I found today (7.4).

**Flags.** ROI for the thesis: INCREMENTAL (a small, citable side
contribution). Cost: about one hour for the issue; two to four hours plus
review time for a pull request. Not a model question, so no agnosticism flag.

## 7.1 What I ran

| Check | Setup | Result |
| --- | --- | --- |
| Reproduction on the latest release **[ran]** | torchao 0.18.0, torch 2.11.0, CPU, a scratch environment; `private/tools/torchao/verify_asym.py` | Defect present |
| Reproduction on upstream main **[ran]** | Source tree at commit `4ab9d61` (2026-11-12) | Identical numbers |
| Fix direction **[ran]** | `private/tools/torchao/verify_fix.py`: compute the parameters with `choose_qparams_affine` instead | Training-time weight equals the deployed weight in 100% of elements, int2/3/4/8 |
| Code reading **[read]** | `IntxFakeQuantizer._per_channel_or_group_forward`, `get_groupwise_affine_qparams`, `_choose_qparams_affine_tinygemm`, `_fake_quantize_per_channel_group` on main | The mechanism in 7.2 |
| Upstream search **[ran]** | GitHub issue search, five queries, 2026-11-13 | No matching issue or pull request |

The test is the one that matters for training: does the weight the model
trains with equal the weight it is deployed with? My reference is a textbook
asymmetric round-to-nearest written from scratch. TorchAO's own deployed path
matches that reference in 100% of elements at every bit width, so the
reference and the library agree on what "correct" is.

## 7.2 The mechanism

For signed integer types with `is_symmetric=False`, the per-group and
per-channel path mixes two conventions **[read]**:

1. It asks a helper for scale and zero point. That helper, with its default
   arguments, uses the convention of one specific 4-bit kernel: an unsigned
   range from 0 to 2^n − 1 and a zero point that is a *real number in weight
   units*.
2. It casts that zero point to a 32-bit integer, which truncates it.
3. It applies the result as an *integer offset* on the *signed* range.

The symmetric path, the per-token path, and the 32-bit branch of the same
function do not do this. The code even carries a note above these lines:
`# TODO: refactor this to use choose_qparams_affine`.

## 7.3 How large the effect is

Worst error of one weight, in quantization steps (a correct rounding stays at
0.5), and relative error of the whole matrix **[ran]**:

| Weights | Bits | Deployed path | Training-time fake quantizer |
| --- | ---: | --- | --- |
| Gaussian, no outlier | 3 | 0.50 steps, 0.217 | 1.38 steps, 0.221 |
| Gaussian, no outlier | 4 | 0.50 steps, 0.101 | 2.39 steps, 0.107 |
| Gaussian, no outlier | 8 | 0.50 steps, 0.006 | 38.4 steps, 0.031 |
| Three outliers added | 3 | 0.50 steps, 0.054 | 8.96 steps, 2.67 |
| Three outliers added | 4 | 0.50 steps, 0.036 | 11.9 steps, 0.73 |
| Three outliers added | 8 | 0.50 steps, 0.012 | 131 steps, 0.50 |

- **It is quiet on well-behaved weights and severe on skewed ones.** When a
  group's range is roughly symmetric, the truncated zero point happens to be
  0 and about 99% of elements still match the deployed weight; only the ends
  of the range are clipped. One large weight in a group breaks the whole
  group. That fits what the 2026-11-01 draft records for your INT3 runs: a KL
  of 6.2 right after preparation, against 0.22 for the deployed INT3 model.
- **It is not an INT3 problem.** It grows with bit width: at 8 bits, even
  without outliers, the matrix error is five times the deployed path's.
- **Symmetric quantization is unaffected:** training-time and deployed
  weights match in 100% of elements in every case.

## 7.4 Is it "a bug" or "an unsupported setting"?

This is the point a maintainer may argue, so it is worth stating fairly
**[read]**:

| For "unsupported" | For "bug" |
| --- | --- |
| The automatic route from a deployment configuration to a training configuration refuses asymmetric weights ("Only symmetric weight mapping is supported") | The explicit configuration accepts the setting without warning and returns wrong weights |
| Upstream tests cover asymmetric only for per-token activations and for the unsigned 4-bit kernel convention | The deployed path supports exactly this setting and is correct |
| The documentation's weight examples are all symmetric | The code marks these lines as to be refactored, and the 32-bit branch already uses the correct routine |

My view: it is a silent wrong result in a setting the library accepts. Either
outcome upstream, a fix or an explicit error, would be an improvement, and
your report would be the cause of it.

Three updates for the draft before posting:

1. **Name the current main commit** (`4ab9d61`, 2026-11-12) instead of
   `b87db29`, and state that the lines are unchanged.
2. **Mention open pull request #5104** ("Support INT16 per-group and
   per-channel fake quantization", opened 2026-11-12). It changes the same
   condition to route 16-bit types through the correct routine and says it
   does so "without changing lower-bit paths". Your fix is the same move for
   the asymmetric case, so it will need a rebase and is easy to motivate.
3. **Add the "quiet on symmetric ranges" finding and the 8-bit numbers.** The
   draft's title ("flips signs") describes the outlier case only; the general
   statement is stronger and more useful to them.

The draft's own "related observation" also reproduces: with the unsigned
4-bit type and a float zero point, the default integer zero-point precision
gives 2.39 steps of error against 0.50 with a float precision.

## 7.5 Recommendation

| Step | Why | Who |
| --- | --- | --- |
| 1. Post the issue with the three updates | It is the citable part and costs an hour | You (an outward action) |
| 2. Offer the fix in the issue, and open the pull request once #5104 is merged or by rebasing onto it | One condition plus a parametrized test; the earlier session had the test failing 8 of 8 before and passing after | You; the Meta contributor agreement is needed |
| 3. Do not wait for a merge | Review timing is outside your control | - |

**Thesis wording.** After step 1: "reported upstream (pytorch/ao issue N)".
"Fixed upstream" only if a pull request is merged. The project's own
deployed-path quantizer stays the method either way, so nothing in the
results depends on this.

**What is where.** The 2026-11-01 draft lived only in a temporary folder; a
copy is now at `private/tools/torchao/issue-draft-2026-11-01.md` beside the two check
scripts. The patch file from that session was already gone from the temporary
folder; the change itself is the diff inside the draft.

## 7.6 Limits of this verification

- CPU only, FP32 inputs in my script. The earlier session also covered BF16.
- The main-branch run used the downloaded source tree on torch 2.11.0, not a
  built development install.
- "No matching issue" means five keyword searches found none. It is not proof
  that none exists.
- I did not run TorchAO's test suite with the fix; the earlier session did
  (90 passed, 135 skipped, 0 failed, by its record).
