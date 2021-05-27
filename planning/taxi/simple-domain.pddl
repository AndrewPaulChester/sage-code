;Header and description

(define (domain navigation)

;remove requirements that are not needed
(:requirements :strips  :typing  :equality)

(:types ;todo: enumerate types and their hierarchy here, e.g. 
vehicle space agent - object
car truck bus - vehicle
 space
 fuelstation taxi passenger - agent
)

; un-comment following line if constants are needed
;(:constants )

(:predicates 
    (in ?a - agent ?s - space)
    (carrying-passenger ?t - taxi ?p - passenger)
    (empty ?t - taxi)
    (destination ?p - passenger ?s - space)
    (delivered ?p - passenger)
    (full-tank ?t)
    (has-money ?t)
)


(:action move
    :parameters (?t - taxi ?s1 - space ?s2 - space)
    :precondition (and (in ?t ?s1)
    )
    :effect (and 
        (not (in ?t ?s1))
        (in ?t ?s2)
        (not (full-tank ?t))
    )
)

(:action pick-up
    :parameters (?t - taxi ?p - passenger ?s - space)
    :precondition (and (in ?t ?s)
                        (in ?p ?s)
                        (empty ?t)
    )
    :effect (and 
        (not (empty ?t))
        (not (in ?p ?s))
        (carrying-passenger ?t ?p)
    )
)

(:action drop-off
    :parameters (?t - taxi ?p - passenger ?s - space)
    :precondition (and (carrying-passenger ?t ?p)
                        (in ?t ?s)
                        (destination ?p ?s)
    )
    :effect (and 
        (not (carrying-passenger ?t ?p))
        (empty ?t)
        (delivered ?p)
        (has-money ?t)
    )
)

(:action refuel
    :parameters (?t - taxi ?f - fuelstation ?s - space)
    :precondition (and (in ?t ?s)
                        (in ?f ?s)
                        (has-money ?t)
    )
    :effect (and 
        (not (has-money ?t))
        (full-tank ?t)
    )
)

)
