# FE method for the glacier problem

## introduction

This file documents the more mathematical aspects of `steady.py`, and of its numerical methods.

The mathematical glacier problem is to determine the steady-state surface elevation, equivalently thickness, of a glacier on land.  The glacier lives in a time-independent climate, and it sits on a time-independent bedrock topography.  The ice flow model is the shallow ice approximation (SIA).

The problem is of free-boundary type because we do not know, in advance of the solution, what area is ice-covered.  We compute the ice volume and ice area simultaneously.

## shallow ice approximation model

There are two fields which are input data:

  * a bed elevation function $z=b(x,y)$, and
  * a surface mass balance (SMB) function $a(x,y,z)$

In default runs done by `steady.py`, $b$ and $a$ are given by synthetic formulas.  However, $b$ can be read from a file with option `-bdata`.

The surface elevation $z=s(x,y)$, and the corresponding thickness $H(x,y)=s(x,y)-b(x,y)$, are the primary unknowns of the model.  Note that the SMB function might depend on the surface elevation $z$: $a=a(x,y,s(x,y))$

To write the model equations we fix an overall flow constant $\Gamma>0$ and a Glen flow law power $n\ge 1$.  For convenience and comparison to related equations we set $p=n+1$.  The traditional glaciology choices are $n=3$ and $p=4$.  In these terms the steady, isothermal, non-sliding SIA equation is:

```math
-\nabla \cdot \left(\Gamma H^{p+1} |\nabla s|^{p-2} \nabla s\right) = a(s) \qquad (1)
```

This PDE resembles two simpler and better-known nonlinear and degenerate diffusion equations, namely the porous media equation and the $p$-Laplacian equation.  Written with a generic unknown $u$ and source $f$, the porous media equation is $-\nabla\cdot (C u^{\gamma-1} \nabla u)=f$ [E10], while the $p$-Laplacian equation is $-\nabla \cdot (C |\nabla u|^{p-2} \nabla u)=f$ [BL93, E10].  Since equation (1) combines aspects of both it is a "doubly-nonlinear" and "doubly-degenerate" diffusion.

Equation (1) is actually the interior condition in a variational inequality (VI) [E10, KS80], or nonlinear complementarity problem, subject to the constraint $s\ge b$, equivalently $H\ge 0$ [JB12].

### power transformation

A doubly-nonlinear equation like (1) was first addressed mathematically by [R70], who introduced a power transformation as an analysis technique.  This transformation was extended to non-flat beds by [JB12].  It converts the SIA equation (1) to a modified $p$-Laplacian equation.

First we define constant powers:

```math
\omega = \frac{p-1}{2p}, \qquad \mu = \frac{p+1}{2p}.
```

(The standard $p=4$ case for glaciology has $\omega=3/8$ and $\mu=5/8$.)  Now define

```math
H = u^\omega
```

This transforms parts of equation (1):

```math
\begin{align*}
H^{p+1} &= u^{(p^2-1)/2p} \\
\nabla s &= \nabla (H + b) = \omega u^{-\mu} (\nabla u + \omega^{-1} u^\mu \nabla b)
\end{align*}
```

The function

$$\Phi(u) = - \omega^{-1} u^\mu \nabla b,$$

is a "tilt" [JB12] because it arises from the bed elevation gradient.  Now

$$|\nabla s|^{p-2} \nabla s = \omega^{p-1} u^{-(p^2-1)/2p} |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))$$

Observe the cancellation between the power on $u$ arising from $H^{p+1}$ versus the power on $u$ in the above expression.

Let $\gamma = \Gamma \omega^{p-1}$.  For the transformed thickness $u$, SIA equation $(1)$ becomes

$$-\nabla \cdot \left(\gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))\right) = a(u^\omega+b) \qquad (2)$$

Equation $(2)$ is a modified form of the $p$-Laplacian equation [JB12].  It is in the degenerate-diffusion $p>2$ case (not the fast-diffusion case $1<p<2$).

There seems to be no literature which really helps us understand understand the effect of the tilt $\Phi(u)$, but see commentary in [JB12].

Regarding consequences of possible dependence of $a$ on the surface elevation, and an understanding of such problems as energy minimization, at least when the bed is flat, see [JRBB11].

### obstacle problem

As noted, PDEs (1) and (2) are to be understood as free-boundary problems.  To state these problems mathematically, and to define weak forms suitable for Firedrake solutions, we need to define admissible sets for the solution variables, and write down variational inequality (VI) forms.

We will solve these free-boundary problems over a fixed map-plane domain $\Omega \subset \mathbb{R}^2$.  For the unknown $s$ we define the admissible set

$$\mathcal{K}_b = \left\{v \in W_b^{1,p}(\Omega) \,:\, v \ge b\right\}$$

Note that $s=b$ on the boundary of the domain, a fixed boundary condition, but that also $s=b$ over an unknown ice-free portion of $\Omega$.  We seek $H$ and $u$ from the admissible set

$$\mathcal{K}_0 = \left\{v \in W_0^{1,p}(\Omega) \,:\, v \ge 0\right\}$$

The variational inequality (VI) weak form corresponding to (1) is to find $s \in \mathcal{K}_b$ so that

```math
\begin{align*}
&\int_\Omega \Gamma H^{p+1} |\nabla s|^{p-2} \nabla s \cdot \nabla(s-r) \\
 &\qquad \ge \int_\Omega a(s) (s-r) \qquad (3)
\end{align*}
```

for all $r\in\mathcal{K}_b$.

Similarly, the VI weak form corresponding to (2) is to find $u\in\mathcal{K}_0$ so that

```math
\begin{align*}
&\int_\Omega \gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \cdot \nabla(u-v) \\
 &\qquad \ge \int_\Omega a(u^\omega+b) (u-v) \qquad (b)
\end{align*}
```

for all $v\in\mathcal{K}_0$.

VI weak form (3) is derived by multiplying(1) by the test function $s-r$ and then integrating by parts [B16].  VI (4) is derived similarly [JB12].

Our primary goal is to find accurate and efficient finite element (FE) solutions of problem (3) and (4).  There are several approaches to doing this, and it is not a settled matter of which works best.

Because of the low regularity of the solution near the free boundary our FE approach uses $s,u\in CG_1$.  We apply a non-mixed primal method, and apply a bounds-adapted Newton iteration [BM06] to handle the several nonlinearities.  This often works acceptably, but it can become fragile, e.g. Newton convergence when the bed elevation has large gradient [B16].

To handle elevation-dependent SMB models in both (3) and (4) we apply a Picard iteration.  For (4) we also optionally apply a Picard iteration on the tilt [JB12].

Such primal methods seem to make mass-conservation errors, essentially of the type discussed in [JAS13], and possibly mitigated by the FE space choices in [B23].  The FE methods considered below only partially address these fragility and conservation issues.

### Picard iteration on the tilted $p$-Laplacian

Supposing $\nabla b\ne 0$, then VIs (3) and (4) are _not_ the first-order conditions for minimizing any functional.  However, in the flat bed case $b=0$ one may regard the problems as minimizations.

To handle (4) by a Picard iteration we abstract and simplify it to give the _tilted $p$-Laplacian obstacle problem_ [JB12].  Let $Z(x,y)$ be a fixed vector field and $A(x,y)$ a fixed source function.  Consider:

$$\int_\Omega \gamma |\nabla u - Z|^{p-2} (\nabla u - Z) \cdot \nabla(u-v) \ge \int_\Omega A (u-v) \qquad (5)$$

for all $v\in\mathcal{K}$.  In (5) the data are the source function $A$ and the tilt $Z$.

Inequality (5) is the first-order condition for minimizing

$$J(u) = \int_\Omega \frac{\gamma}{p} |\nabla u - Z|^p - A u$$

over $\mathcal{K}$ [JB12].  Thus (5) modifies a standard $p$-Laplacian diffusion equation $-\nabla\cdot(|\nabla u|^{p-2} \nabla u) = f$ [BL93] to have a zero-function obstacle, and to make $\nabla u$ want to agree with $Z$, while balancing the source term $A$.

Theorem 3.3 in [JB12] shows that VI problem (5) and the minimization of $J(u)$ are equivalent and well-posed.  However, the SIA VI problem (4) is not in form (5) because of the solution-dependent vector field $Z=\Phi(u)$.  Only existence is known for (4), even when $a$ is independent of $s$ [JB12].  Note that properties of VI problem (3) can be derived from properties of (4) by reversing the power transformation.

In practice, elevation-dependent surface mass balance can generally be included into the above minimization framework.  However, for some dependences $a(s)$, non-existence is known [JRBB11], on a flat bed.  We always Picard iterate to solve $a(s)$ cases.

One way to solve VI problem (4) is to directly apply a VI-adapted Newton solver, specifically the `vinewtonrsls` SNES type from PETSc [BM06].  This works in many circumstances, as long as the initial iterate is nonzero in a roughly reasonable pattern, and as long as grid sequencing is used when seeking high resolution.

In the VI problem (4) the tilt field $Z=\Phi(u)$ is in fact $u$-dependent.  Likewise $A$ may be $u$-dependent.  Thus we put a Picard iterative wrapper around (5), as follows, updating the $Z$ field, and the $A$ source, with each iteration.  That is, we seek iterate $u_k \in \mathcal{K}_0$ from the previous iterate $u_{k-1} \in \mathcal{K}_0$.  Define

```math
Z_k = \Phi(u_{k-1}), \quad A_k = a(u_{k-1}^\omega +b)
```

based on the previous iterate.  We seek $u_k \in \mathcal{K}$ such that, for all $v\in\mathcal{K}$,

```math
\begin{align*}
&\int_\Omega \gamma |\nabla u_k - Z_k|^{p-2} (\nabla u_k - Z_k) \cdot \nabla(u_k-v) \\
&\qquad \ge \int_\Omega A_k (u_k-v) \qquad (6)
\end{align*}
```

This solution method seems to converge well when solved as a primal problem using FE methods.  A stopping criterion for Picard iterations is not clear, and we simply use a fixed number.

Note that solving (6) still requires a VI-adapted Newton solver, because it is an obstacle problem with a nonlinear operator.  However, the problem is close to $p$-Laplacian type.  However, solving (3) or (4) directly by a Newton solver is also implemented and effective.

### References

[BL93] Barrett, J. W. and Liu, W. B. (1993). _Finite element approximation of the $p$-Laplacian_. Mathematics of Computation 61 (204), 523--537

[BM06] Benson, S. and Munson, T. (2006). _Flexible complementarity solvers for large-scale applications_, Optim. Methods Softw. 21, 155--168

[B23] Brinkerhoff, D. J. (2023). _Compatible finite elements for glacier modeling_. Computing in Science & Engineering, 25 (3), 18--28

[B16] Bueler, E. (2016). _Stable finite volume element schemes for the shallow-ice approximation_. Journal of Glaciology, 62 (232), 230--242

[E10] Evans, L. C. (2010). _Partial Differential Equations_, Grad. Stud. Math. 19, 2nd ed., AMS Press, Providence

[JAS13] Jarosch, A. H., Schoof, C. G., and Anslow, F. S. (2013). _Restoring mass conservation to shallow ice flow models over complex terrain_, The Cryosphere 7 (1), 229--240

[JB12] G. Jouvet and E. Bueler (2012). _Steady, shallow ice sheets as obstacle problems: Well-posedness and finite element approximation_, SIAM J. Appl. Math 72 (4), 1292--1314

[JRBB11] G. Jouvet, J. Rappaz, E. Bueler, and H. Blatter (2011). _Existence and stability of steady-state solutions of the shallow-ice-sheet equation by an energy-minimization approach_, Journal of Glaciology, 57 (202), 345--354

[KS80] D. Kinderlehrer and G. Stampacchia (1980). _An Introduction to Variational Inequalities and
their Applications_, Academic Press, New York

[R70] P. A. Raviart (1970). _Sur la resolution de certaines equations paraboliques non lineaires_, J. Funct. Analysis 5, 299--328
