;Header and description

(define (domain craftrooms)

;remove requirements that are not needed
(:requirements :strips  :typing  :equality :fluents)

(:types ;todo: enumerate types and their hierarchy here, e.g. 
 tile
 item
 resource
 player
 direction
 room
)

; un-comment following line if constants are needed
(:constants wooden_pickaxe stone_pickaxe - item;
log stone plank stick - resource
tree rock wall coin none - tile
north south east west - direction)

(:predicates ;todo: define predicates here
    (wooden_pickaxe ?p - player)
    (stone_pickaxe ?p - player)
    (furnace ?p - player)
    (facing ?p - player ?t - tile)
    ;(moved ?p - player ?d - direction)
    ;(cleared ?p - player ?d - direction)
    (adjacent ?r - room ?r - room ?d - direction)
    (in ?p - player ?r - room)
    (door ?r - room ?r - room ?t - tile)
)


(:functions ;todo: define numeric functions here
    (have ?p - player ?r - resource )
    (contains ?r - room ?t - tile )
    (cost)
)

;define actions here


;facing action
(:action face
    :parameters (?p - player ?t - tile ?r - room)
    :precondition (and 
        (in ?p ?r)
        (> (contains ?r ?t) 0)
    )
    :effect (and 
        (forall (?o - tile) (not (facing ?p ?o)))
        (facing ?p ?t)
        (increase (cost) 1)
    )
)

;movement actions
(:action move
    :parameters (?p - player ?d - direction ?r - room ?r2 - room)
    :precondition (and 
                (in ?p ?r)
                (adjacent ?r ?r2 ?d)
                (or (door ?r ?r2 none)
                    (door ?r ?r2 tree)
                    (and (door ?r ?r2 rock)
                        (wooden_pickaxe ?p)
                    )
                )
    )
    :effect (and 
            (forall (?o - tile) (not (facing ?p ?o)))
            (not (in ?p ?r))
            (in ?p ?r2)
            (door ?r ?r2 none)
            (increase (cost) 1)
    )
)

;collecting action

(:action collect-coins
    :parameters (?p - player ?r - room)
    :precondition (and 
                (in ?p ?r)
                (> (contains ?r coin)  0)
    )
    :effect (and 
            (assign (contains ?r coin)  0)
            (increase (cost) 1)
    )
)

; mining actions
(:action mine-tree
    :parameters (?p - player ?r - room)
    :precondition (and (facing ?p tree)
            (in ?p ?r)
        (> (contains ?r tree) 0)
    )
    :effect (and 
        (not (facing ?p tree))
        (decrease (contains ?r tree) 1)
        (increase (have ?p log) 1)
        (increase (cost) 1)
    )
)

(:action mine-rock
    :parameters (?p - player ?r - room)
    :precondition (and (facing ?p rock)
            (in ?p ?r)
                    (> (contains ?r rock) 0)
                    (wooden_pickaxe ?p)
    )
    :effect (and 
        (not (facing ?p rock))
        (decrease (contains ?r rock) 1)
        (increase (have ?p stone) 1)
        (increase (cost) 1)
    )
)

; crafting actions
(:action craft-plank
    :parameters (?p - player)
    :precondition (and 
                    (> (have ?p log)  0)
    )
    :effect (and 
            (decrease (have ?p log) 1)
            (increase (have ?p plank) 4)
            (increase (cost) 1)
    )
)

(:action craft-stick
    :parameters (?p - player)
    :precondition (and 
                    (> (have ?p plank)  0)
    )
    :effect (and 
            (decrease (have ?p plank) 1)
            (increase (have ?p stick) 4)
            (increase (cost) 1)
    )
)

(:action craft-wooden-pickaxe
    :parameters (?p - player)
    :precondition (and 
                    (> (have ?p plank)  2)
                    (> (have ?p stick)  1)
    )
    :effect (and 
            (decrease (have ?p plank) 3)
            (decrease (have ?p stick) 2)
            (wooden_pickaxe ?p)
            (increase (cost) 1)
    )
)

(:action craft-stone-pickaxe
    :parameters (?p - player)
    :precondition (and 
                (> (have ?p stone)  2)
                (> (have ?p stick)  1)
    )
    :effect (and 
            (decrease (have ?p stone) 3)
            (decrease (have ?p stick) 2)
            (stone_pickaxe ?p)
            (increase (cost) 1)
    )
)

)





;clear actions
; (:action clear
;     :parameters (?p - player  ?d - direction)
;     :precondition (and 

;     )
;     :effect (and 
;             (forall (?o - tile) (not (facing ?p ?o)))
;             (forall (?e - direction) (not (moved ?p ?e)))
;             (cleared ?p ?d)
;             (increase (cost) 1)
;     )
; )