#['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', "neutral"  , A, B]

samples =[
        [0,   0,  0.8,0.1,1,0,0,               1, 0.9], #fear/sad
        [1,   0.3,0.2, 0,0.1, 0.3, 0.1,    0.3, 0], #anger
        [0.1, 0, 0.9, 0.1, 0.3, 0.3, 0,   0.25,1] # fear
        [0,   0,0,1,0,0.2, 0.2,              -1, 1] #happy
        [0,   0, 0.2, 0, 0.3, 0.9, 0.2,      0, -1] #surprise -> blue
        [0,   0,0,0.1,0,0, 1,                0.1, 1] #neutral
        [0,   0,]
        ] 