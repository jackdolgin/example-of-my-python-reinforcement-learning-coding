#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Order of things-

First we run mle. We create an `Agent` object, and then update the `Agent`
trial by trial. Finally we extract from `Agent` an attribute called `p_choice`,
which is a vector of the negative log likelihoods from each choice.

    When we create `Agent`, we in turn — on the line starting with
    `self.generate_objs` — create four objects; one object for each of the two
    planets, and one object for each of the two rocket pairs, where one rocket
    pair refers to a first-stage state (when that first-stage state equals
    "one", it means rockets 1 and 2 are on screen (i.e., 1 and 2 comprise the
    values in columns `stim_left` and `stim_right`); when that first-stage
    state equals "three", it means rockets 3 and 4 are on screen).

        Within `self.generate_objs`, we're doing a couple of things. First, we
        create an `Agent` attribute that's a dictionary of the two rocket pairs.
        The keys in the dictionary are the names of the rocket pairs, and the
        values correspond to rocket pair objects (which we generate on the same
        line as when we add it to the dictionary). Thus, we can work within
        `Agent` and whenever we want to access or update a rocket pair object,
        we just call the corresponding key (i.e., the rocket pair name) within
        `self.rocket_pair_objs.` Second, in `self.generate_objs`, we're
        generating another type of object, in th is case planet objects. These
        objects too get created within a dictionary that we call
        `self.planet_objs`.

    After agent is created, we loop through a bunch of trials (this happens on
    the line "for trial in sub_df.itertuples():"). Along the course of a trial,
    we save 27 trial-relevant variables/parameters to `Agent.log`. In terms of
    the sequence of a trial — we use our existing Qmb and Qtd values leftover
    from the previous trial to determine the Q values of each choice option for
    the current trial. Then, we see what participants (or the simulation,
    depending on whether we're running a model or a simluation) chose, and we
    compare that choice with the Q values for that choice to get an estimated
    probability of making that choice according to our model — thus there's
    one probability value per trial, and that creates a vector of probability
    values that gets saved at the end and fed to the minimization function.

    Finally, we update the Qtd and Qmb values based on the reward during that
    trial. See below for a description of the `q_update` function.

        It's worth walking through how the value updates occur. First, let's
        review. There's a Qmb, Qtd, Q, and Qplus for both rocket pairs (e.g.,
        `Qtd(one, red)`). There's also a Q for each planet (e.g., `Q(red)`).
        In this `q_update`` function, we're only going to work with four of
        these Q types:

            1) the Qtd of the combo of the rocket pair we're at and the planet
            we chose (e.g., again, `Qtd(one, red)` if we facing the rocket pair
            that corresponds with first stage state 1 (remember, a rocket pair
            always corresponds to a first stage stage; rockets 1 and 2 appearing
            represent rocket pair 1 aka first stage state 1))

            2 and 3) the Qmb of any combo of the rocket pairs involving the
            planet we chose (e.g., if our destination is the red planet,
            `Qtd(one, red)` and `Qtd(two, red)`). The idea is that a model-based
            update should apply to any choice leading to a planet, whether we
            just made that choice (e.g., making a choice at first stage state 1)
            or even if we didn't (e.g., making a choice at first stage state 2)

            4) the Q value of the planet we arrived at. This one is weird for
            me, because I would think if we chose to go to say the red planet
            and we received a reward, because in this paradigm choosing red
            always takes you to the red planet, there should be no surprises,
            and hence whatever reward you receive on that planet should
            directly feed back to the Qtd (e.g., `Qtd(one, red)`) of the
            first-stage state. Instead, apparently there's this middle layer —
            a q value just for being on a planet (e.g., `Q(red)`).

        Now, let's walk through the steps. It's confusing as I said in 4) above,
        but there's a prediction error, `rpe1`, going from a choice at stage 1
        to the planet destination, even before you learn whether or not you've
        received a reward. That `rpe1` is just the difference between the Q
        value for being on a planet and the Qtd of making the choice to go to
        that planet (again, why is there any prediction error?! you knew you
        were going to end up there!). There's also understandably a prediction
        error going from being on a planet to then finding out what (if
        anything) the reward is; this is `rpe2`. Now we update our Qtd values
        in line with these two prediction errors.

        We actually update the Qtd value for the first-stage choice twice.
        First, we update it to reflect `rpe1`, since that prediction error was a
        direct follow-up from our first-stage choice. However, we also update it
        a second time according to `rpe2`. Notice however that there's a
        `self.λ` term added to the mix. It means that we're reducing how much we
        consider this `rpe2` according to the decimal `self.λ`. We don't factor
        this `rpe2` into the Qtd of the first choice as much as we factor `rpe1`
        because `rpe2` happens later, and it is a sort of indirect, downstream
        consequence of our first-stage choice (even though this first-stage
        choice is deterministically related to the planet you end up at). I
        kind of think of it as a cascading. Your rpes from later on waterfall
        back down to your earlier choices that set up your later choices (and
        the ensuing prediction errors), but because the later rpes are a little
        temporally distanced from your first-stage choice, you don't give as
        much weight to that rpe as `rpe1`.

        Likewise, we also update the Q value of the second-stage (i.e., the
        'choice' at the planet to press spacebar for reward) with `rpe2`.

        Finally, we do our model-based updates for the first-stage, as described
        in the above bullet "2 and 3)". We're updating the first-stage choices
        according to `rpe2`, aka we're setting them equal to the newly-updated
        Q value for the second-stage (planet) we visited during the trial.
        Again, we do this updating for both first-stage states where the choice
        destination is the current planet, even for the first-stage state (e.g.
        `Qmb(two, red)`) we didn't visit during the current trial. That's the
        beauty of being model free! You use your model to update all relevant
        state-action pairs, not just the ones you saw on that trial.


Notice the way variables get carried between functions to avoid global
variables. π, ρ, and ws get passed from the csv of parameters for fitting ->
mle -> Agent -> Agent.generate_objs -> RocketPairObj. α, β, and λ likewise
get passed from the csv of parameters for fitting -> mle -> being an attribute
of Agent. Each iteration we fit the model using the mle function, the ws and the
other six greek letters will change.

"""

import numpy as np
from pandas import options
from random import gauss
from scipy.stats import gamma, norm, beta

options.display.float_format = '{:.2f}'.format # do I need this?

class RocketPairObj:
    def __init__(self, pair, planets, π, ρ, ws):
        self.name = pair
        self.planets = planets
        self.π = π
        self.ρ = ρ
        self.w_dict = dict(zip(('high', 'faux_high', 'faux_low', 'low'), ws))   # we set the four values in this dict equal to the four ws we pass in from the csv of parameters for fitting
        self.Qtd = {planet: .5 for planet in self.planets}
        self.Qmb = {planet: .5 for planet in self.planets}
        self.Q = {planet: .5 for planet in self.planets}
        self.Qplus = {planet: .5 for planet in self.planets}

    def q_integrate(self, prev_planet, stake, pair_sides, prev_pair_sides, prev_rocket_pair):

        w = self.w_dict[stake]

        for planet in self.planets:
            new_q = self.Qmb[planet] * w + self.Qtd[planet] * (1 - w)
            self.Q[planet] = self.Qplus[planet] = new_q

            if prev_planet == planet and prev_rocket_pair == self.name:
                self.Qplus[planet] += self.π
            if (prev_planet == planet) == (prev_pair_sides == pair_sides) and prev_planet is not None:
                self.Qplus[planet] += self.ρ

    def qmb_update(self, planet_objs):
        for planet in self.planets:
            self.Qmb[planet] = planet_objs[planet].Q


class PlanetObj:
    def __init__(self):
        self.Q = .5
        self.rwalk_min = -4
        self.rwalk_max = 5
        self.rwalk_mean = 0
        self.rwalk_sd = 2
        self.treasure = (self.rwalk_min + self.rwalk_max) / 2
        self.random_walk()

    def random_walk(self):
        rwalk_change = gauss(self.rwalk_mean, self.rwalk_sd)

        self.treasure = round(self.treasure + rwalk_change)

        while not self.rwalk_min <= self.treasure <= self.rwalk_max:
            if self.treasure > self.rwalk_max:
                self.treasure -= 2 * (self.treasure - self.rwalk_max)
            elif self.treasure < self.rwalk_min:
                self.treasure += 2 * (self.rwalk_min - self.treasure)


class Agent:

    '''
    It's worth emphasizing that even though in this script the only output of
    `Agent` that we use is the dictionary values within the key `p_choice`
    within the `Agent.log` attribute, `Agent.log` stashes a value for each trial
    for each of 27 variables (keys); those are — `trial_index`, `og_pair`,
    `pair_sides`, `stake_type` `Qtd(one,red)`, `Qtd(one,purple)`,
    `Qmb(one,red)`, `Qmb(one,purple)`, `Q(one,red)`, `Q(one,purple)`,
    `Qplus(one,red)`, `Qplus(one,purple)`, `Qtd(two,red)`, `Qtd(two,purple)`,
    `Qmb(two,red)`, `Qmb(two,purple)`, `Q(two,red)`, `Q(two,purple)`,
    `Qplus(two,red)`, `Qplus(two,purple)`, `Q(red)`, `Q(purple)`, `p_choice`,
    `planet`, `points`, `rpe1`, and `rpe2`
    '''

    def __init__(self, procedure, rocket_pairs, planets, α, β, λ, π, ρ, ws):
        self.planets = planets
        self.rocket_pair_objs = {}
        self.planet_objs = {}
        self.generate_objs(rocket_pairs, π, ρ, ws)
        self.procedure = procedure
        self.α = α
        self.β = β
        self.λ = λ
        self.prev_rocket_pair, self.prev_pair_sides, self.prev_planet, self.planet = [None] * 4
        self.log = {}

    def trial(self, trial_index, rocket_pair, stake, pair_sides,
              preset_planet=None, trial_was_completed=True, preset_payoff=None):

        if trial_was_completed:

            self.log_var(
                ("trial_index", trial_index),
                ("og_pair", rocket_pair),
                ("pair_sides", pair_sides),
                ("stake_type", stake)
            )

            self.log_qs()

            rocket_pair_obj = self.rocket_pair_objs[rocket_pair]                # this line says, "Let's work with the rocket pair object that matches the rocket pair that was seen in this trial (we know which one it is since it was fed into `Agent.trial` via the parameter `rocket_pair`)"

            # Adds together (integrates) Qtd and Qmd — which have been
            # calculated most recently on the previous trial — thereby giving
            # us a Qplus value for choosing each planet given the fact..........
            rocket_pair_obj.q_integrate(                                        # ...
                self.prev_planet,                                               # ... 1) we're at the given planet — this affects wehther to add the ρ parameter into Qplus and whether to add the π parameter into Qplus
                stake,                                                          # ... 2) we're facing the given stakes — this affects which w parameter to use (w_high, w_faux_high, w_faux_low, or w_low)
                pair_sides,                                                     # ... 3) the rocket pair is presented with each rocket on the side it happens to be on (and not flipped the opposite way) — this affects whether to add the ρ parameter into Qplus
                self.prev_pair_sides,                                           # ... 4) we're coming off having just completed a trial with the rockets presented on the sides they were on — this affects whether to add the ρ parameter into Qplus
                self.prev_rocket_pair                                           # ... 5) we're coming off having just completed a trial with the rocket pairs that happened to be presented, whether they were the same or different as the ones seen now — this affects whether to add the π parameter into Qplus
            )

            self.planet_selection(rocket_pair_obj, preset_planet)

            self.q_update(rocket_pair_obj, preset_payoff)

        else:

            self.planet = preset_planet

        self.remaining_updates(rocket_pair, pair_sides)

    def planet_selection(self, rocket_pair_obj, preset_planet):

        # softmax
        qs = rocket_pair_obj.Qplus.values()                                     # something useful is, the way we're setting up qs, the choices in `weighted_choices` correspond in order to the planets in `self.planets`; the reason how — weighted_choices comes from qs, which comes from Qplus in the rocket pair object, and we set up Qplus...
        qs = np.fromiter((Q for Q in qs), float)                                # ... using "for planet in self.planets"; therefore, qs is in the same order as `self.planets`, which comes from having fed in `planets` when we defined that rocket pair object; we also fed `planets` into `Agent` when we created `self.planets`
        weighted_choices = np.exp(qs * self.β)

        weighted_choices = weighted_choices / weighted_choices.sum()            # at this point, we have assigned a probability to each potential decision, based on their Qplus values and the output of those Qplus values after being inputted into a softmax

        if self.procedure == "simulate":
            self.planet = np.random.choice(self.planets, p=weighted_choices)    # if we're in simulation mode, we pick a planet based on the weights from `weighted_choices`
        elif self.procedure == "model":
            self.planet = preset_planet                                         # if we're in modeling mode, we 'pick' whatever planet the participant picked during that trial

        p_choice = weighted_choices[self.planets.index(self.planet)]            # whatever planet we've picked, we extract its index within `self.planets` (i.e., the order that planet appears in the list `self.planets`); we then use that index to pull the corresponding weight of choosing that planet from `weighted_choices`

        self.log_var(
            ("p_choice", p_choice),
            ("planet", self.planet)
        )

    # note that this update occurs after we've chosen a planet
    # we update after after the choice, but we don't use those values
    # until choice on next trial; update results in starting
    def q_update(self, rocket_pair_obj, preset_payoff):

        planet_obj = self.planet_objs[self.planet]

        rpe1 = planet_obj.Q - rocket_pair_obj.Qtd[self.planet]

        if self.procedure == "simulate":
            payoff = self.planet_objs[self.planet].treasure
        elif self.procedure == "model":
            payoff = preset_payoff

        rpe2 = payoff - planet_obj.Q

        rocket_pair_obj.Qtd[self.planet] += rpe1 * self.α
        planet_obj.Q += rpe2 * self.α
        rocket_pair_obj.Qtd[self.planet] += rpe2 * self.α * self.λ

        for loops_rocket_pair in self.rocket_pair_objs.values():
            loops_rocket_pair.qmb_update(self.planet_objs)

        self.log_var(
            ("points", payoff),
            ("rpe1", rpe1),
            ("rpe2", rpe2)
        )

    def remaining_updates(self, rocket_pair, pair_sides):

        [planet.random_walk() for planet in self.planet_objs.values()]

        make_present_the_past = (("prev_rocket_pair", rocket_pair),
         ("prev_pair_sides", pair_sides),
         ("prev_planet", self.planet))

        for param in make_present_the_past:
            setattr(self, param[0], param[1])

    def log_var(self, *args):
        for arg in args:
            key, val = arg
            if key not in self.log.keys():
                self.log[key] = [val]
            else:
                self.log[key].append(val)

    def generate_objs(self, rocket_pairs, π, ρ, ws):

        for pair in rocket_pairs:
            self.rocket_pair_objs[pair] = RocketPairObj(pair, self.planets, π, ρ, ws)

        for planet in self.planets:
            self.planet_objs[planet] = PlanetObj()

    def log_qs(self):
        for rocket_pair_name, rocket_pair_obj in self.rocket_pair_objs.items():
            for q_type in ("Qtd", "Qmb", "Q", "Qplus"):
                for planet_name, q_for_planet in getattr(rocket_pair_obj, q_type).items():

                    key = q_type + '(' + rocket_pair_name + ',' + planet_name + ')'

                    if key not in self.log.keys():
                        self.log[key] = [q_for_planet]
                    else:
                        self.log[key].append(q_for_planet)

        for planet_name, planet_obj in self.planet_objs.items():

            key = 'Q(' + planet_name + ')'

            if key not in self.log.keys():
                self.log[key] = [planet_obj.Q]
            else:
                self.log[key].append(planet_obj.Q)


def mle(params, rocket_pairs, planets, sub_df, include_priors):
    print(*params[0:5])
    print(params[5:])
    agent = Agent("model", rocket_pairs, planets, *params[0:5], params[5:])     # the star in front of `params[0:5]` means we treat each value within `params[0:5]` as independent from one another, as opposed to within a list. They correspond to α, β, λ, π, ρ. Meanwhile, `params[5:]` corresponds to the four starting ws for each of the four stake groups (high, faux_high, faux_low, and low), repsectively.

    for trial in sub_df.itertuples():
        agent.trial(trial.trial_index, trial.og_pair, trial.stake_type, trial.pair_sides,
                    trial.preset_planet, trial.completed_trial, trial.points)

    probs = agent.log['p_choice']

    if include_priors:

        for index, param in enumerate(params):
            if index == 1:
                prior_prob = gamma.pdf(param, 3, scale=.2)

            elif index in {3, 4}:
                prior_prob = norm.pdf(param, 0, 1)

            else:
                prior_prob = beta.pdf(param, 2, 2)

            probs.append(prior_prob)

    aposteriori = -np.log(probs).sum()
    print("----------------")
    print(agent.log)
    return aposteriori
