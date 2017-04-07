# Caterpillar Tube Pricing

Dis mashine learning, it kool

## Simple model

If bracket: cost = f(base_tube_cost, quantity)

* base_tube_cost = f(diameter, wall, length, num_bends, bend_radius) - linear regression from cost of 1-bracket tubes
* cost coefficients - linear regression of cost*quantity over quantity

If non-bracket: cost = base tube cost

* base tube cost = f(diameter, wall, length, num_bends, bend_radius) - linear regression from cost of 1-non-bracket tubes