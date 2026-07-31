# FE method for the glacier problem

## introduction

This Markdown file documents the more mathematical aspects of `steady.py`, and its numerical methods.

The mathematical glacier problem is to determine the steady-state surface elevation, equivalently thickness, of a glacier in a fixed time-independent climate and on a fixed bedrock topography.  The ice flow model is the shallow ice approximation (SIA).

## shallow ice approximation model

There are two fields which are input data:

  * a bed elevation function $z=b(x,y)$, and
  * a surface mass balance function $a(x,y,z),$ which might depend on surface elevation $z$.

In default runs done by `steady.py`, $b$ and $a$ are given by synthetic formulas.  However, $b$ can be read from a file with option `-bdata`.

Note that the surface elevation $z=s(x,y)$ and the corresponding thickness $H(x,y)=s(x,y)-b(x,y)$ are primary unknowns of the model.

Fix an overall flow constant $\Gamma>0$ and a Glen flow law power $n\ge 1$.  Let $p=n+1$.  The traditional case for glaciology has $n=3$ and $p=4$.

Now the steady, isothermal, non-sliding SIA equation is:

```math
-\nabla \cdot \left(\Gamma H^{p+1} |\nabla s|^{p-2} \nabla s\right) = a(s) \qquad (1)
```

This PDE resembles two simpler and better-known nonlinear and degenerate diffusion equations, the porous media equation and the $p$-Laplacian equation.  Written with generic unknown $u$ and source $f$, the porous media equation is $-\nabla\cdot (C u^{\gamma-1} \nabla u)=f$ [E10].  The $p$-Laplacian equation is $-\nabla \cdot (C |\nabla u|^{p-2} \nabla u)=f$ [BL93, E10].  Since equation (1) combines aspects of both of these equations it is known as a "doubly-nonlinear" and "doubly-degenerate" diffusion [R70].

Equation (1) is actually the interior condition in a variational inequality (VI) [E10] or nonlinear complementarity problem (NCP) subject to the constraint $s\ge b$, equivalently $H\ge 0$ [JB12].

### power transformation

The following transformation, introduced in [R70], and discussed and extended to non-flat beds in [JB12], converts the SIA equation (1) to a $p$-Laplacian equation.

For this transformation we define constant powers:

```math
\omega = \frac{p-1}{2p}, \qquad \phi = \frac{p+1}{2p}.
```

The standard $p=4$ case for glaciology has $\omega=3/8$ and $\phi=5/8$.

Now define

```math
H = u^\omega
```

The following calculations allow us to transform parts of equation (1):

```math
\begin{align*}
H^{p+1} &= u^{(p^2-1)/2p} \\
\nabla s &= \nabla (H + b) = \omega u^{-\phi} (\nabla u + \omega^{-1} u^\phi \nabla b)
\end{align*}
```

Let

$$\Phi(u) = - \omega^{-1} u^\phi \nabla b$$

Note that [JB12] calls $\Phi(u)$ the "tilt" because it arises from the bed elevation gradient.  Thus

$$|\nabla s|^{p-2} \nabla s = \omega^{p-1} u^{-(p^2-1)/2p} |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))$$

Observe the cancellation between the power on $u$ arising from $H^{p+1}$ versus the power on $u$ in the above expression.

Let $\gamma = \Gamma \omega^{p-1}$.  Now, for the transformed thickness $u$, SIA equation $(1)$ becomes

$$-\nabla \cdot \left(\gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))\right) = a(u^\omega+b) \qquad (2)$$

Equation $(2)$ is a modified form of the $p$-Laplacian equation [JB12], in the degenerate $p>2$ case.  There does not seem to be a literature which really helps us understand understand the effect of the tilt $\Phi(u)$, but see commentary in [JB12].

Regarding consequences of dependence of $a$ on the surface elevation, if any, and an understanding of such problems as energy minimization, at least when the bed is flat, see [JRBB11].

### obstacle problem in transformed thickness

Now define the admissible set

$$\mathcal{K} = \left\{v \in W_0^{1,p}(\Omega) \,:\, v \ge 0\right\}$$

The VI weak form corresponding to $(2)$ is to find $u\in\mathcal{K}$ so that

```math
\begin{align*}
&\int_\Omega \gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \cdot \nabla(u-v) \\
 &\qquad \ge \int_\Omega a(u^\omega+b) (u-v) \qquad (3)
\end{align*}
```

for all $v\in\mathcal{K}$.

This weak form VI (3) is derived, as usual, by multiplying $(2)$ by the test function $u-v$, and integrating by parts [JB12].

Our primary goal is an accurate finite element (FE) solution of problem $(3)$.  However, there are several approaches to doing this, and it is not a settled matter of which works best.

The most direct method of solving $(3)$ by FE is to use $u\in CG_1$, as a non-mixed primal method, and apply a bounds-adapted Newton iteration [BM06] to handle the several nonlinearities.

This default method used here is primal and adds a Picard iteration on the tilt, described below and as in [JB12].  This often work acceptably, but it can become fragile, especially with regard to Newton convergence when the bed elevation has large gradient [B16].

Such primal methods seem to make mass-conservation errors, essentially of the type discussed in [JAS13], and possibly mitigated by the FE space choices in [B23].  The FE methods considered below only partially address these fragility and conservation issues.

### tilted $p$-Laplacian objective

Supposing $\nabla b\ne 0$, then VI (3) is _not_ the first-order condition for minimizing any functional.  We can abstract and simplify (3), to give the _tilted $p$-Laplacian obstacle problem_ [JB12].  Let $Z(x,y)$ be a fixed vector field and $A(x,y)$ a fixed source function.  Consider:

$$\int_\Omega \gamma |\nabla u - Z|^{p-2} (\nabla u - Z) \cdot \nabla(u-v) \ge \int_\Omega A (u-v) \qquad (4)$$

for all $v\in\mathcal{K}$.  In (4) the data are the source function $A$ and the tilt $Z$.

Inequality (4) is the first-order condition for minimizing

$$J(u) = \int_\Omega \frac{\gamma}{p} |\nabla u - Z|^p - A u$$

over $\mathcal{K}$ [JB12].  Thus (4) modifies a standard $p$-Laplacian diffusion equation $-\nabla\cdot(|\nabla u|^{p-2} \nabla u) = f$ [BL93] to have a zero-function obstacle, and to make $\nabla u$ want to agree with $Z$, while balancing the source term $A$.

Theorem 3.3 in [JB12] shows that VI problem (4) and the minimization of $J(u)$ are equivalent and well-posed.  However, the SIA VI problem (3) is not in form (4) because of the solution-dependent vector field $Z=\Phi(u)$.

Only existence is known for (3), even when $a$ is independent of $s$ [JB12].

In practice, elevation-dependent surface mass balance can generally be included into the above minimization framework.  However, for some dependences $a(s)$, non-existence is known [JRBB11], on a flat bed.

### Picard iteration

One way to solve VI problem (3) is to directly apply a VI-adapted Newton solver, specifically the `vinewtonrsls` SNES type from PETSc [BM06].  This works in many circumstances, as long as the initial iterate is nonzero in a roughly reasonable pattern, and as long as grid sequencing is used.

In the VI problem (3) the tilt field $Z=\Phi(u)$ is in fact $u$-dependent.  Likewise $A$ may be $u$-dependent.  An option for solving the SIA with these dependencies is to Picard iterate on solving (4) [JB12].  That is, we may put an iterative wrapper around (4), updating the $Z$ field, and the $A$ source, with each iteration.

To describe this in formulas, we seek iterate $u_k \in \mathcal{K}$ from previous iterate $u_{k-1} \in \mathcal{K}$.  Define

```math
Z_k = \Phi(u_{k-1}), \quad A_k = a(u_{k-1}^\omega +b)
```

based on the previous iterate.  We seek $u_k \in \mathcal{K}$ such that, for all $v\in\mathcal{K}$,

```math
\begin{align*}
&\int_\Omega \gamma |\nabla u_k - Z_k|^{p-2} (\nabla u_k - Z_k) \cdot \nabla(u_k-v) \\
&\qquad \ge \int_\Omega A_k (u_k-v) \qquad (5)
\end{align*}
```

This solution method seems to converge well when solved as a primal problem using FE methods.  A stopping criterion for Picard iterations is not clear, and we simply use a fixed number.

Note that solving (5) still requires a VI-adapted Newton solver, because it is an obstacle problem with a nonlinear operator.  However, the problem is close to $p$-Laplacian type, versus solving (3) directly by a Newton solver.

However, with $u \in CG_1$ in this primal form, mass conservation issues remain when and where $\nabla b$ is large in magnitude [B23, JAS13].  Fixing this is a longer-term project.

### References

[BL93] Barrett, J. W. and Liu, W. B. (1993). _Finite element approximation of the $p$-Laplacian_. Mathematics of Computation 61 (204), 523--537

[BM06] Benson, S. and Munson, T. (2006). _Flexible complementarity solvers for large-scale applications_, Optim. Methods Softw. 21, 155--168

[B23] Brinkerhoff, D. J. (2023). _Compatible finite elements for glacier modeling_. Computing in Science & Engineering, 25 (3), 18--28

[B16] Bueler, E. (2016). _Stable finite volume element schemes for the shallow-ice approximation_. Journal of Glaciology, 62 (232), 230--242

[E10] Evans, L. C. (2010). _Partial Differential Equations_, Grad. Stud. Math. 19, 2nd ed., AMS Press, Providence

[JAS13] Jarosch, A. H., Schoof, C. G., and Anslow, F. S. (2013). _Restoring mass conservation to shallow ice flow models over complex terrain_, The Cryosphere 7 (1), 229--240

[JB12] G. Jouvet and E. Bueler (2012). _Steady, shallow ice sheets as obstacle problems: Well-posedness and finite element approximation_, SIAM J. Appl. Math 72 (4), 1292--1314

[JRBB11] G. Jouvet, J. Rappaz, E. Bueler, and H. Blatter (2011). _Existence and stability of steady-state solutions of the shallow-ice-sheet equation by an energy-minimization approach_, Journal of Glaciology, 57 (202), 345--354

[R70] P. A. Raviart (1970). _Sur la resolution de certaines equations paraboliques non lineaires_, J. Funct. Analysis 5, 299--328
