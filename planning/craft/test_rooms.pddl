

    (define (problem test_problem) (:domain craftrooms)
    (:objects 
        
        p - player
        r1 r2 r3 r4 - room
    )

    (:init
        ;todo: put the initial state's facts and numeric values here
       
        (= (cost) 0)
        
        ;player facts
        (= (have p log) 0)
        (= (have p plank) 0)
        (= (have p stone) 0)
        (= (have p stick) 0)
        (not (wooden_pickaxe p))
        (not (stone_pickaxe p))(not (moved p c0 c0))

        (in p r1)

        ;room layout
        (adjacent r1 r2 east)
        (adjacent r2 r1 west)
        (adjacent r3 r4 east)
        (adjacent r4 r3 west)

        (adjacent r1 r3 south)
        (adjacent r3 r1 north)
        (adjacent r2 r4 south)
        (adjacent r4 r2 north)
        ;room contents
        
        (= (contains r1 tree) 2)
        (= (contains r1 rock) 0)
        (= (contains r1 coin) 0)

        (= (contains r2 tree) 0)
        (= (contains r2 rock) 0)
        (= (contains r2 coin) 2)

        (= (contains r3 tree) 0)
        (= (contains r3 rock) 4)
        (= (contains r3 coin) 0)

        (= (contains r4 tree) 0)
        (= (contains r4 rock) 0)
        (= (contains r4 coin) 5)

        ;doors
        (door r1 r2 none)
        (door r1 r3 tree)
        (door r2 r4 rock)
        (door r3 r4 rock)

        (door r2 r1 none)
        (door r3 r1 tree)
        (door r4 r2 rock)
        (door r4 r3 rock)

        ; (not (cleared p north))
        ; (not (cleared p south))
        ; (not (cleared p east))
        ; (not (cleared p west))
        ;(= (capacity t) 100)
    )

    (:goal (and
            ;(> (have p log) 0)
            (in p r4)
            ;(wooden_pickaxe p)
        )
    )

    ;un-comment the following line if metric is needed
    (:metric minimize (cost))
    )
    
