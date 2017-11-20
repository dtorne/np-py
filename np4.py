#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:03:33 2017

@author: david
TODO There is an error. Filters too much.
"""

from pyspark.sql import Row
from pyspark.sql.types import LongType, ArrayType
from pyspark.sql.functions import array, explode, lit, col, udf
from pyspark.sql import *


import sys

from pyspark import SparkContext, SparkConf



def parseNPFunction(path):
    import re
    pattern = re.compile("[-]?\\d")
    
    numVar = 0
    numClauses = 0
    clauses = set()
    file = open(path, 'r') 
    for line in file:
        lineA = line.split()
        if len(lineA) == 0:
            continue
        
        if lineA[0] == "p":
            numVar = int(lineA[2])
            numClauses = int(lineA[3])
            
        elif pattern.match(lineA[0]):
            clause = set(map(lambda x : int(x), lineA[:-1]))
            if len(clause) > 0:
                clauses.add(frozenset(clause))
    
    numClauses = len(clauses)
    return (numVar, numClauses, clauses)



"""
    minimclause (outdated comment)

    Reduces the number of clauses to the minimum relevant where newVar appears (rest have been already checked)
    @param clauses Array of sets. Each set is a clause of type (int, int int...)
    @param level . How many variable to check. From 1 to level + the new variable
    @param newVariable. the new variable whose valid clause add to the existing
    Requires level < newVariable.
"""
def minimClause(clauses, level, newVariable, onlynewVariable = True):
    varsFilter = set(range(1,level+1))
    varsFilter.add(newVariable)
    minClauses = set()
    for clause in clauses:
        isMinimum = True
        hasnewVariable = False
        for item in clause:
            if abs(item) not in varsFilter:
                isMinimum = False
                break
            elif abs(item) == newVariable:
                hasnewVariable = True
        if isMinimum and ((not onlynewVariable) or hasnewVariable):
            minClauses.add(clause)
    return minClauses

def minimClauseAllVars(clauses, level, onlynewVariables = True):
    varsFilter = set(range(1,level+1))
    minClauses = set()
    for clause in clauses:
        isMinimum = False
        hasOneNew = False
        for item in clause:
            if abs(item) in varsFilter:
                isMinimum = True
            else:
                hasOneNew = True
        if isMinimum and ((not onlynewVariables) or hasOneNew):
                minClauses.add(clause)
                
    return minClauses



#Returns True if exist solution, False otherwise
def checkBoolean(clauses, instanceSet):
    for clause in clauses:
        isSolution = False
        for item in clause:
            if item not in instanceSet:
                isSolution = True
        if not isSolution:
            return isSolution
    return True

"""
Calculates if this instantiation passes the closes.
Returns status. 
-1- delete. passes neither 0 nor 1 newVar instance for this id instance
1- 01b passes newVar 0 
2- 10b passes newVar 1
3- 11b passes newVar 0 and 1
Accpetance condition level < newVariable. Otherwise unexpected behaviour
"""
def updateState(clauses, idValue, level, numVar, initStatus = 3):
       
    idSet = set()
    for i in range(0,level):
        if (idValue >> i) % 2 == 1:
            idSet.add(i+1)
        else:
            idSet.add(-(i+1))
    
    finalStatus = 0
    #Since it goes from final to newVar = level+1, updates status with the final case level+1
    #The rest of variables are used to check if it, the instance, can be eleminated. status = -1
    for newVariable in range(numVar,level,-1):
        status = initStatus
        #Check if 0 for newVariable was passing. If status 1 or 3
        if status % 2 == 1:
            idSet.add(-newVariable)
            passes0 = checkBoolean(clauses, instanceSet = idSet)
            idSet.remove(-newVariable)
            if not passes0:
                #newVariable at 0 does not pass. Status then 2 or 0
                status = status - 1
            #For the rest of variables than level+1, we just want to see if it can be eliminated with both passes0 and passes1
            #So we continue directly to next iteration if we know in this newVariable there is solution
            elif newVariable != level+1:
                continue
  
        #Check if 1 for newVariable was passing. Status 2 or 3
        if (status>>1) % 2 == 1:
            idSet.add(newVariable)
            passes1 = checkBoolean(clauses, instanceSet = idSet)
            if not passes1:
                #newVariable at 1 does not pass. Status then 1 or 0
                status = status % 2
            idSet.remove(newVariable)
        if status == 0:
            return -1
        else:
            finalStatus = status
    
    return finalStatus


def expandLevel(idCol,stateCol,newVariable):
    result = []
    #We add newVar = 0
    if stateCol % 2 == 1:
        result.append(idCol)
    if (stateCol>>1) % 2 == 1:
        result.append(2**(newVariable-1)+idCol)
    return result


def solveCNF(path="", startLevel=10):
    
    # 20 Vars - uf20-01.cnf # 4726 sol. 7714 nodes
    # 50 Vars - uf50-05.cnf
    
    if path == "":
        path = path

    expandids = udf(expandLevel, ArrayType(LongType()))

    #graphDF = sc.parallelize([])
    (numVar, numClauses, clauses) = parseNPFunction(path)     
    
    initData = [i for i in range(0,2**(startLevel))]
    initRDD = sc.parallelize(initData)
    initDF = initRDD.map(lambda x: Row(id=long(x), state=3))
    
    walkerDF = sqlContext.createDataFrame(initDF).cache()
    #graphDF = walkerDF.select("id").withColumn("level",lit(startLevel))

    for newVar in range(startLevel+1, numVar+1):
        if newVar == startLevel+1:
            reducedClause = minimClauseAllVars(clauses, newVar-1, onlynewVariables = False)
        else:
            reducedClause = minimClauseAllVars(clauses, newVar-1)
        
        #Parse state with clauses
        walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateState(clauses = reducedClause, idValue = row['id'], level = newVar-1, numVar = numVar))).toDF().where(col('state') != -1)
        walkerDF = walkerDF.withColumn("id",explode(array(expandids(col("id"),col("state"),lit(newVar))))).withColumn("state",lit(3)).select(explode("id").alias("id"),"state")
        #graphDF = graphDF.union(walkerDF.select("id").withColumn(lit(newVar)))
        walkerDF.select("id").write.save("50output"+ str(newVar),format="csv")
        
    return walkerDF.select("id")


conf = SparkConf().setAppName("Boolean Solver")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
graphDF = solveCNF("uf50-05.cnf", 15).cache()
print("Count: ",graphDF.count())
