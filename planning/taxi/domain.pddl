;Header and description

(define (domain navigation)

;remove requirements that are not needed
(:requirements :strips  :typing  :equality)

(:types ;todo: enumerate types and their hierarchy here, e.g. 
vehicle space agent - object
car truck bus - vehicle
 space
 taxi passenger - agent
)

; un-comment following line if constants are needed
;(:constants )

(:predicates ;todo: define predicates here
    (in ?a - agent ?s - space)
    (above ?s1 - space ?s2 - space)
    (left ?s1 - space ?s2 - space)
    (carrying-passenger ?t - taxi ?p - passenger)
    (empty ?t - taxi)
    (destination ?p - passenger ?s - space)
    (delivered ?p - passenger)

)


;(:functions ;todo: define numeric functions here
;
;)

;define actions here
(:action move-up
    :parameters (?t - taxi ?s1 - space ?s2 - space)
    :precondition (and (in ?t ?s1)
                        (above ?s2 ?s1)
    )
    :effect (and 
        (not (in ?t ?s1))
        (in ?t ?s2)
    )
)

(:action move-down
    :parameters (?t - taxi ?s1 - space ?s2 - space)
    :precondition (and (in ?t ?s1)
                        (above ?s1 ?s2)
    )
    :effect (and 
        (not (in ?t ?s1))
        (in ?t ?s2)
    )
)
(:action move-left
    :parameters (?t - taxi ?s1 - space ?s2 - space)
    :precondition (and (in ?t ?s1)
                        (left ?s2 ?s1)
    )
    :effect (and 
        (not (in ?t ?s1))
        (in ?t ?s2)
    )
)
(:action move-right
    :parameters (?t - taxi ?s1 - space ?s2 - space)
    :precondition (and (in ?t ?s1)
                        (left ?s1 ?s2)
    )
    :effect (and 
        (not (in ?t ?s1))
        (in ?t ?s2)
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
    )
)

)
