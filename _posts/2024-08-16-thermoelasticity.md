---
layout: post
title: "The interaction between heat and elasticity"
author: "Alessandro Morita"
categories: posts
tags: [physics,mechanics,thermodynamics]
image: stress-distribution-bracket.jpg
---

When subject to an increase in temperature, a solid object will usually expand, and this expansion will act to increase internal stresses. In this way, variations in temperature affect solids' mechanical properties.

On the other hand, inside a solid body, heat transfer occurs mostly by diffusion from hot to colder zones. This depends not only on the temperature gradients, but also on the body's geometry. Hence, the thermal conductivity and static elasticity problem are coupled.

In many multiphysics applications, one decouples this system by solving it sequentially:
* First solving the heat equation, thus obtaining a temperature profile $T(x)$;
* Then plugs in this temperature to a thermal stress term in the linear elastic equations, and solves them for the displacement field $u$, which can then be used to derive strains, stresses etc.

In this article we justify why one may decouple the problem in this form in the specific case of solids. We will mostly follow the classical text [1] of Landau & Lifschitz. Throughout the text we employ the Einstein convention of summing over repeated indices. 

# Recap: the stress tensor and free energy

We follow [1], in which the primary quantity that we consider is the **Helmholtz free energy** of a solid under deformation. We present the rationale for choosing Helmholtz free energy below.

> Tl:DR: 
> The Helmholtz free energy appears as a shortcut to obtain the stress tensor. If we can write $F = F(\varepsilon_{ij})$, i.e. the free energy as a function of strain, then the stress tensor can be calculated immediately by differentiation, i.e. $F$ is the generating function of stress and strain.

By considering virtual displacements, it is not hard to show that the work done by the system if it contains stresses and strains is , *per unit volume*

$$\delta W=-\sigma_{ij} \delta \varepsilon_{ij}$$

where $\sigma_{ij}$ is the stress tensor and $\varepsilon_{ij}$ is the infinitesimal strain tensor. This equation generalizes the term $\delta W = p dV$ for work when we only consider pressure.

Then, for a reversible process, we have the first law of thermodynamics in the form 

$$dU = T dS - \delta W=TdS +\sigma_{ij}d\varepsilon_{ij}$$

if there are no other work terms; here, $U$ is the internal energy and $S$ the entropy, both per unit volume. The Helmholtz free energy per unit volume $F = U - TS$ can then be introduced, and its differential is 

$$dF = - SdT + \sigma_{ij} d\varepsilon_{ij}\tag{1}$$

from which one immediately obtains 

$$\boxed{\sigma_{ij} = \left(\frac{\partial F}{\partial \varepsilon_{ij}}\right)_T}.$$

Thus, if one can express $F$ as a function of the (kinematic) variable $\varepsilon_{ij}$. then we can obtain the (dynamic) stress tensor by pure differentiation.

## Linear elasticity at constant reference temperature

One obtains (*isotropic*) linear elasticity by suitably expanding the free energy in terms of the strain tensor $\varepsilon_{ij}$.

First, we fix a **reference temperature** $T_0$ such that we can set the body to be **undeformed** at this temperature. All quantities in this section are assumed to be evaluated at $T=T_0$; this will be dropped in the following sections.

Since $\sigma_{ij} = \partial F/\partial \varepsilon_{ij}$, there must not be any terms linear in $\varepsilon_{ij}$ in the expansion of $F$, for if there were, then $\sigma_{ij}$ would not be zero. 

Now we consider higher order terms. Since $F$ is a scalar, all the terms in the expansion must be scalars, meaning they must be (1) number-valued functions and (2) invariant under rotations. For a symmetric matrix $A$ there are two rotation-independent quadratic invariants: $\mathrm{Tr\,}(A^2)$ and $(\mathrm{Tr} A)^2$; since $\varepsilon_{ij}$ is symmetric, we assume with no loss of generality that 

$$F(\varepsilon) = F_0 + \frac \lambda 2(\mathrm{Tr}\,\varepsilon)^2+\mu\,\mathrm{Tr}(\varepsilon^2) + \text{higher order terms}$$

or, in index notation

$$F(\varepsilon) = F_0 + \frac \lambda 2 (\varepsilon_{ii})^2 + \mu \varepsilon_{ij} \varepsilon_{ij} + O(\varepsilon^3)$$

$\lambda$ and $\mu$ are, of course, the Lamé coefficients, in units of pressure. We see that they naturally appear as expansion coefficients for the Helmholtz free energy. Notice that, by straightforward differentiation, one finds 

$$\sigma_{ij} = \frac{\partial F}{\partial \varepsilon_{ij}}=2\mu \varepsilon_{ij}+\lambda \varepsilon_{kk} \delta_{ij}$$

which is the usual constitutive law for isotropic linear elasticity.

## Adding temperature

If we set a temperature $T \neq T_0$, we expect there to be deformations even in the absence of external forces. Hence, a linear term in $\varepsilon_{ij}$ must appear in the free energy expansion. Requiring isotropy, the only possible scalar is its trace $\varepsilon_{ii}$, thus $F$ must contain a term of the form 
$$a(T) \varepsilon_{ii}.$$
Assuming temperature variations around $T_0$ to be small, we can expand the coefficient $a(T)$ to first order,

$$F(\varepsilon, T)=F_0(T){-K\alpha (T-T_0) \varepsilon_{ii}} +  \frac \lambda 2 (\varepsilon_{ii})^2 + \mu \varepsilon_{ij} \varepsilon_{ij} + O(\varepsilon^3, (T-T_0)^2)\tag{2}$$

Above, we chose to write the coefficient of $\varepsilon_{ii}$, with no loss of generality, as the bulk modulus $K$ times a factor $\alpha$ which is yet to be determined. This is a convenient choice since $K$ has units of pressure, and the free energy per unit volume has units of energy/volume = pressure as well. Since $\varepsilon_{ii}$ is unitless, we conclude $\alpha$ has units of inverse temperature. 

We have also chosen the constants $\lambda, \mu$ to not depend on $T$; if they did, since they cannot have $O(T-T_0)$ dependence but only $O((T-T_0)^2)$, their contributions would only come at higher orders and thus can be ignored.

Again, by differentiation, we find 

$$\boxed{\sigma_{ij}(T)=-K \alpha (T-T_0) \delta_{ij}+2\mu \varepsilon_{ij}+\lambda \varepsilon_{kk} \delta_{ij} {+ O(\varepsilon^2, (T-T_0)^2)}}\tag{3}$$

To understand the effect of this new thermal stress term, assume a body not subject to external forces, just undergoing a change in temperature. The body will deform, so $\varepsilon \neq 0$, but there are no stresses, hence $\sigma = 0$. Setting $\sigma_{ij} = 0$ above, we can solve for $\varepsilon$ by contracting indices:

$$0=-3K \alpha(T-T_0) + (3\lambda+ 2\mu)\varepsilon_{kk}$$
$$\Rightarrow \varepsilon_{kk}=\alpha(T-T_0)$$

using that $K = \lambda + 2\mu/3$. Recalling that $\varepsilon_{kk} = \nabla\cdot u$ is the relative volume increase due to displacement, we see that $\alpha$ is the (volumetric) thermal expansion coefficient of the body, which we expect to be positive in order for $\varepsilon_{kk}$ and $T-T_0$ to have the same sign.

## How the linear elastic equation changes 

Recall the equations for a linear elastic body are 

$$\rho \frac{\partial^2 u}{\partial t^2}=\mathrm{div}\, \sigma+ f$$

where $f$ is a bulk force (per unit volume). This equation must be supplemented by initial and boundary conditions.

By writing out $\sigma$ as in Eq. (3), we can expand this equation as

$$\boxed{\frac{2(1+\nu)}{E}\rho \frac{\partial^2 u}{\partial t^2}= \Delta u+\frac{1}{1-2\nu} \nabla(\nabla\cdot u)-{\frac{2 \alpha}{3} \frac{1+\nu}{1-2\nu} \nabla T}  + \frac{2(1+\nu)}{E}f}\tag{4}$$

where $E, \nu$ are Young's modulus and Poisson's ratio, respectively, and $u$ is the displacement field. The $\nabla T$ denotes how temperature gradients affect the linear elastic equation.

We must complement this equation with one for the temperature field, i.e. the heat equation. 

## Deriving the heat equation

In solids, in contrast to fluids, internal convection is not an efficient mechanism for heat transfer. Ignoring radiation, whose effects are usually small, we see that heat diffusion is the only mechanism to consider in the *interior* of a solid body (this is not the case for the body's interface with an external medium, like a fluid -- convection plays an important role here, but this enters as a boundary condition).

Hence, heat flux can be written from Fourier's law as 

$$q'' = -k\nabla T$$

where the heat flux vector $q''$ has units of power per unit area, i.e. energy per unit area per unit second, and $k$ is the body's thermal conductivity. Assume a volume $\Omega$ is at a lower temperature than its surroundings; then, it will absorb heat. From the divergence theorem, the total heat absorbed per unit time is, *in the absence of a volumetric source/sink term*,

$$\frac{\delta Q}{dt} = \int_\Omega \nabla\cdot(k\nabla T)\,dx$$

The right-hand side is positive since $\nabla T$ points outward, and the divergence of such a field is positive. 

> If we wanted to consider a heat source, we would need to add a $\int_\Omega \dot q dx$ on the RHS, where $\dot q$ has units of power per unit volume. This term would be added to the RHS of equations (5), (7) and (9).

We can write the left-hand side as a function of entropy per unit volume as 

$$\frac{\delta Q}{dt}=\int_\Omega T\frac{\partial S}{\partial t} dx,$$

from which we derive

$$T \frac{\partial S}{\partial t}=\nabla\cdot(k\nabla T)\tag{5}$$

Now, we want to get rid of entropy to find an equation for $T$. This can be done as follows. First, remember that, from Eq. (1), we had 

$$dF = -SdT + \sigma_{ij} d\varepsilon_{ij};$$

it follows that 

$$S = -\left(\frac{\partial F}{\partial T}\right)_\varepsilon$$

so we can compute the entropy directly from the free energy. Using Eq. (2), which we repeat here ignoring higher-order terms, 

$$F(\varepsilon, T)=F_0(T)-K\alpha (T-T_0) \varepsilon_{ii} + + \frac \lambda 2 (\varepsilon_{ii})^2 + \mu \varepsilon_{ij} \varepsilon_{ij}$$

we immediately find

$$\boxed{S(T) =S_0(T)  +K \alpha \varepsilon_{ii}}\tag{6}$$

where $S_0 \equiv -\partial F_0/\partial T$. Entropy increases as the body expands, which is intuitive. Substituting this into Eq. (5) yields

$$T \frac{\partial S_0}{\partial t}+ K \alpha T \frac{\partial(\nabla\cdot u) }{\partial t}=\nabla\cdot(k\nabla T).$$

where we explicitly wrote $\varepsilon_{ii}$ in terms of the displacement field $u$. 

To further simplify this equation, we need a few ingredients. First, [Mayer's relation](https://en.wikipedia.org/wiki/Relations_between_heat_capacities#Relations) in the form 

$$c_p -c_V=\frac{T}{\rho}\frac{\alpha^2}{\beta}$$

where $\alpha$ is the thermal expansion coefficient and $\beta = 1/K$ is called the isothermal compressibility. We can rewrite this as 

$$K\alpha T = \frac{\rho(c_p-c_V)}{\alpha}$$

which is the term multiplying $\nabla\cdot u$ in the equation; hence

$$T \frac{\partial S_0}{\partial t} + \frac{\rho (c_p - c_V)}{\alpha}  \frac{\partial}{\partial t}\nabla \cdot u = \nabla\cdot(k\nabla T).\tag{7}$$

We note that

$$\frac{\partial S_0}{\partial t}=\frac{\partial S_0}{\partial T} \frac{\partial T}{\partial t}.\tag{8}$$

Now, recall the following fact from thermodynamics. At constant volume, one can relate heat and temperature variation as 

$$\delta Q = C_V dT$$

or, per unit volume, and writing $\delta Q = T dS$,

$$T dS = \rho c_V dT \quad \Rightarrow\quad \left(\frac{\partial S}{\partial T}\right)_V=\frac{\rho c_V}{T}.$$

At constant volume, i.e. when $\varepsilon_{ii} = 0$, $S$ is just $S_0$ (from Eq. (6)) so we can finally rewrite Eq. (8) as 

$$\frac{\partial S_0}{\partial t} = \frac{\rho c_V}{T} \frac{\partial T}{\partial t}$$

and thus we obtain the full heat equation from (7):

$$\boxed{\rho c_V\frac{\partial T}{\partial t} + \frac{\rho (c_p - c_V)}{\alpha} \frac{\partial}{\partial t}(\nabla \cdot u) = \nabla\cdot(k\nabla T)}.\tag{9}$$

# Putting both equations together

Below, we repeat equations (4) and (9), with a little massaging:

$$\begin{align*}
\frac{2(1+\nu)}{E}\rho \frac{\partial^2 {u}}{\partial t^2} &=  \Delta {u}+\frac{1}{1-2\nu} \nabla(\nabla\cdot {u}) \\
&-\frac{2 \alpha}{3} \frac{1+\nu}{1-2\nu} \nabla {T} +\frac{2(1+\nu)}{E}f
\end{align*}$$

$$\rho c_V\frac{\partial {T}}{\partial t} + \frac{\rho (c_p - c_V)}{\alpha} \frac{\partial}{\partial t}(\nabla \cdot {u}) = \nabla\cdot(k\nabla {T})$$

From the presence of $u$ and $T$ in both equations, we see that the two systems are **coupled** and, in principle, would need to be solved jointly.

# What about decoupling?

## Case 1: static solutions

Let us assume steady state, where both $u$ and $T$ have no time dependence. Then, all time derivatives vanish and we are left with

$$ \nabla\cdot(k\nabla {T})=0$$

$$ \Delta {u}+\frac{1}{1-2\nu} \nabla(\nabla\cdot {u})-\frac{2 \alpha}{3} \frac{1+\nu}{1-2\nu} \nabla {T} +\frac{2(1+\nu)}{E}f$$

Conveniently, we have no more $u$ dependence in the heat equation, which can then be solved directly and its result can be plugged into the linear elastic equation as an effective additional force. This justifies the approach we usually see, i.e. first solving the heat equation and then using its result as an additional "force" in the linear elastic equation.

## Case 2: comparing scales

The coupling appears inside the heat equation via the term

$$ \frac{\rho (c_p - c_V)}{\alpha} \frac{\partial}{\partial t}(\nabla \cdot  u)$$

which, as we will argue, *is usually small*: $c_p$ and $c_V$ are very close for solids, and so this term is often completely neglected.

First, from the fact that $\nabla\cdot u = \alpha (T-T_0) \approx \alpha dT$, we conclude that the two terms in the LHS of the heat equation are

$$\begin{align*}
\text{First term:} & \quad \rho c_V \frac{\partial T}{\partial t}\\
\text{Second term:} &\quad \frac{\rho(c_p-c_V)}{\alpha} \frac{\partial}{\partial t}(\nabla\cdot u) \approx \rho(c_p-c_V)\frac{\partial T}{\partial t} \approx T \frac{\alpha^2}{\beta} \frac{\partial T}{\partial t} 
\end{align*}$$

Our comparison then becomes on the scales of 

$$\rho c_V \quad\text{vs.}\quad T \frac{\alpha^2}{\beta}$$

Let us compare these terms:
* [Reference](https://www.engineeringtoolbox.com/volum-expansion-coefficients-solids-d_1894.html) for $\alpha$ values (in $10^{-6} K^{-1}$)
* [Reference](https://www.knowledgedoor.com/2/elements_handbook/isothermal_compressibility.html) for $\beta$ values (at 300 K, in $\mathrm{GPa^{-1}}$)
* [Reference](https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities) for $\rho c_V$ (in $\mathrm{J/K.cm^3}$)

For some common materials at 300 K, we see that indeed the term multiplying $\nabla\cdot u$ is around < 5% of the first one:

| Material | $\alpha$ (in $10^{-6} K^{-1}$) | $\beta$ (in  $\mathrm{GPa^{-1}}$) | $\rho c$ (in $\mathrm{J/K.cm^3}$) | $\rho c$ (in SI units) | $T \alpha^2/\beta$ (in SI units) | ratio |
| -------- | ------------------------------ | --------------------------------- | --------------------------------- | ---------------------- | -------------------------------- | ----- |
| Aluminum | 69.0                           | 0.01385                           | 2.422                             | 2,422,000              | 103,126                          | 4%    |
| Copper   | 49.9                           | 0.0073                            | 3.45                              | 3,450,000              | 102,329                          | 3%    |

Because of this, we can mostly neglect this term and stay with the decoupled heat equation

$$\rho c_V\frac{\partial {T}}{\partial t}=  \nabla\cdot(k\nabla {T})$$

whose result can then be plugged back into the linear elastic equation.

# Conclusion

We have seen that, although theoretically coupled, when considering small displacements and small variations in temperature, we can rewrite thermoelastic coupling into a sequential coupling where the heat equation is solved first, and its result is fed into the linear elastic equation. 

# References

[1]L. Landau, L. Pitaevskii, E. Lifshitz, and A. Kosevich. *Theory of Elasticity. Course of Theoretical Physics Volume 7*. Butterworth-Heinemann, 3 edition, (1986)
