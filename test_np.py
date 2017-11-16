#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:03:33 2017

@author: david
"""
import np
newVar = 19
currentLevel = 18
(numVar, numClauses, clauses) = np.parseNPFunction(path="uf20-01.cnf")


# Test
test = clauses.pop()
assert(len(test.intersection(set([0]))) == 0)
clauses.add(test)
assert(numClauses > numVar)

minClauses = np.minimClause(clauses, currentLevel, newVar)
#Includes the newVar in all clauses
hasNewVariables = True
#Has newVar positive clauses
hasPosSign = False
#Has newVar negative clauses
hasNegSign = False
for clause in minClauses:
    if newVar not in clause and -newVar not in clause:
        hasNewVariables = False
    if newVar in clause:
        hasPosSign = True
    elif -newVar in clause:
        hasNegSign = True
        
assert(hasNewVariables)
assert(hasPosSign and hasNegSign)

minClauses = np.minimClause(clauses, currentLevel, newVar, False)
for clause in minClauses:
    if newVar not in clause and -newVar not in clause:
        hasNewVariables = False
    if newVar in clause:
        hasPosSign = True
    elif -newVar in clause:
        hasNegSign = True
        
assert(not hasNewVariables)
assert(hasPosSign and hasNegSign)

minClause = minimClause(clauses,level=8,newVariable=9)
#Based on minClause = 
#{frozenset({9, -5, -1}),
# frozenset({-8, 4, 7}), 
#frozenset({9, 6, 1}),
# frozenset({-8, 4, -9}),
# frozenset({1, -5, -9})}
#Value 100000001 only var1 at 1 and newVar=9 positive is not blocked by any
#Value 000000001 all negative included newVar=-9 is blocked by {1, -5, -9}
# So status should be 10b which is 2
status2 = updateStatus(clauses, idValue=1, status=3, level=8, newVariable=9)
assert(status2==2)
#Value 100000000 only newVar=9 positive is blocked by {9, -5, -1}
#Value 000000000 all negative included newVar=-9 is not blocked by any
# So status should be 01b which is 1
status1 = updateStatus(clauses, idValue=0, status=3, level=8, newVariable=9)
assert(status1==1)