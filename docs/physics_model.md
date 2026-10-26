# docs/physics_model.md

## Governing Equations

1. **Momentum Equation**
   
   \[ \rho (1 - D) \frac{d \mathbf{u}}{dt} = -\nabla p + \nabla \cdot \boldsymbol{\tau} + \mathbf{f}\]

2. **Energy Equation**
   
   \[ \frac{d}{dt} (\rho h) = -\nabla \cdot (\mathbf{u} h) + \nabla \cdot (k \nabla T) + Q \]

3. **Constitutive Models**
   - Johnson–Cook strain hardening
   - Cowper–Symonds strain‑rate effect
   - Adiabatic shear plug (Recht‑Ipson)

4. **Damage Accumulation**
   
   \[ \dot{D} = \frac{\sigma}{\sigma_{0}} \]

(The full implementation details live in `simulation.py` and the constitutive‑model modules.)
