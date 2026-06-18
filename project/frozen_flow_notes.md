Frozen Flow
===========
Most stable when dropping features entirely: p_drop=1
Less stable when injecting all feature features: p_drop=0
Mostly trainable at: p_drop = 0.20
Untrainable at .50/.50

Next up:
-ChatGPT:
    -read test of response
    -ask: do I really want to tune with 100% injection? Wouldn't that tune for an overly-reliant network?

Key
===
A->A
C->B
D->C
E-A -> D (Ref D)
E-B -> D-V1 (early stopping)
F->    D-V2 (wider sino padding)
G->    D-V3 (wider sino padding)
H->    D-V4 (dropout)

I1-A-> E       (Ref E)
I1-B-> E-V1      (250 epochs)
E1-C-> E-V2      (125 epochs)
J   -> E-V3      (fill = 1)
L   -> E-V4
M   -> E-V5

K (omit)
N-R (leave alone)