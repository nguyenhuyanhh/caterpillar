# Caterpillar Tube Pricing

Dis mashine learning, it kool

## Simple model

If bracket: cost = base tube cost * bracket quantity multiplier

* base tube cost = f(diameter, wall, length, num_bends, bend_radius) - linear regression from cost of 1-bracket tubes
* bracket quantity multiplier = f(quantity) - power law, linear regression through log(quantity)/price

If non-bracket: cost = base tube cost

* base tube cost = f(diameter, wall, length, num_bends, bend_radius) - linear regression from cost of 1-non-bracket tubes