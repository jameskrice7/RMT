/-!
# Machine-checked identities for "Spurious Information"

Core Lean 4 only (no mathlib).  Run from this directory with

    lean RMTChecks.lean

The file checks
1. the combinatorics of the semicircle and Marchenko--Pastur moments
   (Catalan and Narayana numbers against the recursions implied by the
   self-consistent Stieltjes equations), exactly, up to order ten;
2. the telescoping structure behind the exact spurious-information formula,
   its nonnegativity for any antitone sequence (e.g. digamma terms), and its
   strict growth when uninformative coordinates are added;
3. the algebra of the BBP and canonical-correlation phase transitions;
4. the information-rate algebra for the symmetric matrix Ornstein--Uhlenbeck
   process and its eigenvalue (Dyson) projection;
5. finite-sample Stein/KL and portfolio-risk identities.

Analytic facts (Wishart moments, digamma limits, Girsanov) are proved in the
paper; Lean certifies the algebraic and combinatorial steps that connect them.
-/

namespace RMT

/-! ## 1. Moment combinatorics -/

/-- Binomial coefficients by Pascal's rule. -/
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

/-- Integer polynomials in `q`, as coefficient lists (constant term first). -/
abbrev Poly := List Int

def Poly.add : Poly → Poly → Poly
  | [], g => g
  | f, [] => f
  | a :: f, b :: g => (a + b) :: Poly.add f g

def Poly.scale (c : Int) (f : Poly) : Poly := f.map (c * ·)

/-- Multiply by `q`. -/
def Poly.shift (f : Poly) : Poly := 0 :: f

def Poly.mul : Poly → Poly → Poly
  | [], _ => []
  | a :: f, g => Poly.add (Poly.scale a g) (Poly.shift (Poly.mul f g))

/-- Remove trailing zeros so that equal polynomials have equal lists. -/
def Poly.norm (f : Poly) : Poly := (f.reverse.dropWhile (· == 0)).reverse

/-- Marchenko--Pastur moments from the self-consistent equation
    `h = 1 + x h (1 - q + q h)`:  `m_k = (1-q) m_{k-1} + q Σ_{j<k} m_j m_{k-1-j}`. -/
def mpMoments : Nat → List Poly
  | 0 => [[1]]
  | k + 1 =>
    let ms := mpMoments k
    let prev := ms.getD k []
    let conv := (List.range (k + 1)).foldl
      (fun acc j => Poly.add acc (Poly.mul (ms.getD j []) (ms.getD (k - j) []))) []
    ms ++ [Poly.add (Poly.mul [1, -1] prev) (Poly.shift conv)]

def mpMoment (k : Nat) : Poly := Poly.norm ((mpMoments k).getD k [])

/-- Narayana closed form `m_k = Σ_{j<k} (1/k) C(k,j) C(k,j+1) q^j`. -/
def narayana (k j : Nat) : Int := ((choose k j * choose k (j + 1)) / k : Nat)

def narayanaPoly (k : Nat) : Poly :=
  if k = 0 then [1] else (List.range k).map (fun j => narayana k j)

theorem mp_moments_are_narayana :
    (List.range 11).all (fun k => mpMoment k == narayanaPoly k) = true := by
  decide +kernel

/-- The first four Marchenko--Pastur moments, as quoted in the paper. -/
theorem mp_low_moments :
    mpMoment 1 = [1] ∧ mpMoment 2 = [1, 1] ∧ mpMoment 3 = [1, 3, 1] ∧
      mpMoment 4 = [1, 6, 6, 1] := by
  decide +kernel

/-- Narayana numbers sum to Catalan numbers: the `q = 1` moments are the
    moments of the squared semicircle. -/
def catalan : Nat → Nat
  | 0 => 1
  | n + 1 => (List.range (n + 1)).foldl
      (fun acc j => acc + catalanAux j * catalanAux (n - j)) 0
where catalanAux : Nat → Nat
  | j => choose (2 * j) j / (j + 1)

def catalanClosed (n : Nat) : Nat := choose (2 * n) n / (n + 1)

theorem catalan_recursion_matches :
    (List.range 13).all (fun n => catalan n == catalanClosed n) = true := by
  decide +kernel

theorem narayana_rows_sum_to_catalan :
    (List.range 11).all (fun k =>
      ((narayanaPoly k).foldl (· + ·) 0) == (catalanClosed k : Int)) = true := by
  decide +kernel

/-- Semicircle moments: `μ_{2k} = Cat_k` from the Wigner recursion
    `μ_{2k+2} = Σ_j μ_{2j} μ_{2k-2j}` (non-crossing pair partitions). -/
def wignerEven : Nat → List Nat
  | 0 => [1]
  | k + 1 =>
    let ms := wignerEven k
    ms ++ [(List.range (k + 1)).foldl (fun acc j => acc + ms.getD j 0 * ms.getD (k - j) 0) 0]

theorem semicircle_moments_catalan :
    (List.range 13).all (fun k => (wignerEven 12).getD k 0 == catalanClosed k) = true := by
  decide +kernel

/-! ## 2. Spurious information: telescoping, sign and monotonicity -/

/-- `sumTo c n = c 0 + ... + c (n-1)`. -/
def sumTo (c : Nat → Rat) : Nat → Rat
  | 0 => 0
  | n + 1 => sumTo c n + c n

theorem sumTo_add (c : Nat → Rat) (p r : Nat) :
    sumTo c (p + r) = sumTo c p + sumTo (fun i => c (p + i)) r := by
  induction r with
  | zero => show sumTo c p = sumTo c p + 0; grind
  | succ r ih =>
    show sumTo c (p + r) + c (p + r) = sumTo c p + (sumTo (fun i => c (p + i)) r + c (p + r))
    rw [ih]; grind

theorem sumTo_sub (c d : Nat → Rat) (r : Nat) :
    sumTo (fun i => c i - d i) r = sumTo c r - sumTo d r := by
  induction r with
  | zero => show (0 : Rat) = 0 - 0; grind
  | succ r ih => simp only [sumTo, ih]; grind

theorem sumTo_nonneg (c : Nat → Rat) (h : ∀ i, 0 ≤ c i) (r : Nat) : 0 ≤ sumTo c r := by
  induction r with
  | zero => simp [sumTo]
  | succ r ih =>
    have := h r
    simp only [sumTo]; grind

theorem sumTo_pos (c : Nat → Rat) (h0 : ∀ i, 0 ≤ c i) (h : ∀ i, 0 < c i) (r : Nat)
    (hr : 0 < r) : 0 < sumTo c r := by
  cases r with
  | zero => contradiction
  | succ r =>
    have a := sumTo_nonneg c h0 r
    have b := h r
    simp only [sumTo]; grind

/-- With `c i = ψ((n-i)/2)`, twice the spurious information is
    `S(p) + S(r) - S(p+r)` where `S(k) = Σ_{i<k} c i`. -/
def twiceBias (c : Nat → Rat) (p r : Nat) : Rat := sumTo c p + sumTo c r - sumTo c (p + r)

/-- Telescoping identity (Proposition on exact spurious information). -/
theorem twiceBias_eq (c : Nat → Rat) (p r : Nat) :
    twiceBias c p r = sumTo (fun i => c i - c (p + i)) r := by
  unfold twiceBias
  rw [sumTo_add, sumTo_sub]; grind

/-- Nonnegativity for any antitone sequence (digamma is increasing, so
    `i ↦ ψ((n-i)/2)` is antitone). -/
theorem twiceBias_nonneg (c : Nat → Rat) (anti : ∀ i j, i ≤ j → c j ≤ c i) (p r : Nat) :
    0 ≤ twiceBias c p r := by
  rw [twiceBias_eq]
  apply sumTo_nonneg
  intro i
  have := anti i (p + i) (by omega)
  grind

/-- Strict positivity for a strictly antitone sequence and nonempty blocks. -/
theorem twiceBias_pos (c : Nat → Rat) (anti : ∀ i j, i < j → c j < c i) (p r : Nat)
    (hp : 0 < p) (hr : 0 < r) : 0 < twiceBias c p r := by
  rw [twiceBias_eq]
  apply sumTo_pos _ _ _ r hr
  · intro i
    have := anti i (p + i) (by omega)
    grind
  · intro i
    have := anti i (p + i) (by omega)
    grind

/-- Adding `m` uninformative coordinates to the second block raises the
    expected plug-in information by an explicit nonnegative amount. -/
theorem twiceBias_noise (c : Nat → Rat) (p r m : Nat) :
    twiceBias c p (r + m) - twiceBias c p r =
      sumTo (fun i => c (r + i) - c (p + r + i)) m := by
  unfold twiceBias
  rw [sumTo_add c r m, show p + (r + m) = (p + r) + m by omega,
    sumTo_add c (p + r) m, sumTo_sub]
  grind

theorem twiceBias_noise_pos (c : Nat → Rat) (anti : ∀ i j, i < j → c j < c i)
    (p r m : Nat) (hp : 0 < p) (hm : 0 < m) :
    twiceBias c p r < twiceBias c p (r + m) := by
  have h := twiceBias_noise c p r m
  have pos : 0 < sumTo (fun i => c (r + i) - c (p + r + i)) m := by
    apply sumTo_pos _ _ _ m hm
    · intro i
      have := anti (r + i) (p + r + i) (by omega)
      grind
    · intro i
      have := anti (r + i) (p + r + i) (by omega)
      grind
  grind

/-- Symmetry of the spurious information in the two block dimensions. -/
theorem twiceBias_symm (c : Nat → Rat) (p r : Nat) : twiceBias c p r = twiceBias c r p := by
  unfold twiceBias; rw [Nat.add_comm p r]; grind

/-! ## 3. Phase-transition algebra -/

/-- Marchenko--Pastur edges `(1 ± s)^2`, `q = s^2`: sum and product. -/
theorem mp_edges (s : Rat) :
    (1 + s) ^ 2 + (1 - s) ^ 2 = 2 * (1 + s ^ 2) ∧
      (1 + s) ^ 2 * (1 - s) ^ 2 = (1 - s ^ 2) ^ 2 := by
  constructor <;> grind

/-- BBP: at the threshold `ℓ = 1 + s` the outlier `ℓ(1 + q/(ℓ-1))`
    sits exactly at the upper Marchenko--Pastur edge. -/
theorem bbp_outlier_at_threshold (s : Rat) (hs : s ≠ 0) :
    (1 + s) * (1 + s ^ 2 / ((1 + s) - 1)) = (1 + s) ^ 2 := by
  grind

/-- BBP: the squared-overlap numerator vanishes at the threshold. -/
theorem bbp_overlap_at_threshold (s : Rat) (hs : s ≠ 0) :
    1 - s ^ 2 / ((1 + s) - 1) ^ 2 = 0 := by
  grind

/-- BBP inversion: an outlier `λ = ℓ + qℓ/(ℓ-1)` satisfies
    `ℓ^2 - (λ + 1 - q) ℓ + λ = 0`, the quadratic used to infer `ℓ` from data. -/
theorem bbp_inverse (ell q : Rat) (h : ell - 1 ≠ 0) :
    let lam := ell + q * ell / (ell - 1)
    ell ^ 2 - (lam + 1 - q) * ell + lam = 0 := by
  intro lam
  grind

/-- Canonical-correlation transition: if `τ^2 (1-c₁)(1-c₂) = c₁ c₂`
    (threshold `τ = t_c^2`), the outlier formula
    `τ (1 - c₁ + c₁/τ)(1 - c₂ + c₂/τ)` equals `u^2 + w^2 + 2uw` with
    `u^2 = c₁(1-c₂)`, `w^2 = c₂(1-c₁)`, `uw = τ(1-c₁)(1-c₂)`,
    i.e. the upper Wachter edge `(u+w)^2`. -/
theorem cca_outlier_at_threshold (c1 c2 τ : Rat) (hτ : τ ≠ 0)
    (thr : τ * τ * ((1 - c1) * (1 - c2)) = c1 * c2) :
    τ * (1 - c1 + c1 / τ) * (1 - c2 + c2 / τ) =
      c1 * (1 - c2) + c2 * (1 - c1) + 2 * (τ * ((1 - c1) * (1 - c2))) := by
  grind

/-- At `t = 1` the canonical-correlation outlier is at one (perfect correlation). -/
theorem cca_outlier_at_one (c1 c2 : Rat) :
    1 * (1 - c1 + c1 / 1) * (1 - c2 + c2 / 1) = 1 := by
  grind

/-! ## 4. Information rate of matrix Ornstein--Uhlenbeck dynamics -/

/-- Entrywise Girsanov rate: `N(N-1)/2` off-diagonal entries with diffusion
    variance `1/N` and stationary variance `1/(2aN)`, and `N` diagonal entries
    with diffusion variance `2/N` and stationary variance `1/(aN)`. -/
def mouRateEntries (N a b : Rat) : Rat :=
  (1 / 2) * (b - a) ^ 2 *
    (N * (N - 1) / 2 * ((1 / (2 * a * N)) / (1 / N)) + N * ((1 / (a * N)) / (2 / N)))

/-- Eigenvalue (Dyson) coordinates: `N` coordinates with diffusion `2/N` and
    `E Σ λ_i^2 = E tr H^2 = (N+1)/(2a)`. -/
def mouRateEigen (N a b : Rat) : Rat := (1 / 2) * (b - a) ^ 2 * ((N + 1) / (2 * a)) / (2 / N)

theorem mou_rate_closed (N a b : Rat) (hN : N ≠ 0) (ha : a ≠ 0) :
    mouRateEntries N a b = N * (N + 1) * (b - a) ^ 2 / (8 * a) := by
  unfold mouRateEntries
  grind

theorem mou_eigen_closed (N a b : Rat) (hN : N ≠ 0) (ha : a ≠ 0) :
    mouRateEigen N a b = N * (N + 1) * (b - a) ^ 2 / (8 * a) := by
  unfold mouRateEigen
  grind

/-- The eigenvalue projection loses no information about the mean-reversion
    rate (data-processing equality). -/
theorem mou_eigen_sufficient (N a b : Rat) (hN : N ≠ 0) (ha : a ≠ 0) :
    mouRateEigen N a b = mouRateEntries N a b := by
  rw [mou_rate_closed N a b hN ha, mou_eigen_closed N a b hN ha]

/-- Stationary second moment: `N(N-1)` off-diagonal terms of variance
    `1/(2aN)` and `N` diagonal terms of variance `1/(aN)` give `(N+1)/(2a)`. -/
theorem mou_trace_moment (N a : Rat) (hN : N ≠ 0) (ha : a ≠ 0) :
    N * (N - 1) * (1 / (2 * a * N)) + N * (1 / (a * N)) = (N + 1) / (2 * a) := by
  grind

/-! ## 5. Finite-sample Stein/KL and portfolio identities -/

/-- `N n/(n-N-1) - N = N(N+1)/(n-N-1)`: the trace part of the expected Stein
    loss reduces to the parameter count over the residual degrees of freedom. -/
theorem stein_trace (N n : Rat) (h : n - N - 1 ≠ 0) :
    N * n / (n - N - 1) - N = N * (N + 1) / (n - N - 1) := by
  grind

/-- Out-of-sample over in-sample minimum-variance risk at finite sample:
    `((n-1)/(n-N)) / ((n-N+1)/n)`, and its proportional limit `(1-q)^{-2}`
    is obtained from the same expression at `n = 1/ε`, `N = q/ε`. -/
theorem gmv_ratio (N n : Rat) (h1 : n - N ≠ 0) (h2 : n - N + 1 ≠ 0) (hn : n ≠ 0) :
    ((n - 1) / (n - N)) / ((n - N + 1) / n) = n * (n - 1) / ((n - N) * (n - N + 1)) := by
  grind

/-- Value of information in the CARA--Gaussian model (scalar case): with prior
    variance `V` and posterior variance `v`, the determinant `1 + B A` with
    `A = 1/v`, `B = V - v` equals `V/v`, and `A⁻¹ + B = V`. -/
theorem cara_scalar (V v : Rat) (hv : v ≠ 0) :
    1 + (V - v) * (1 / v) = V / v ∧ 1 / (1 / v) + (V - v) = V := by
  constructor <;> grind

end RMT
