#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:03:33 2017

@author: david
NP Solver where solves the problem by adding variables, filtering soutions in a graph manner.
For every variable added, goes to root of graph / tree to prune branches of solutions traversing the graph in breath
depth manner, and expands the new solutions of the new variable, repeating the proces to all variables of the problem.
TODO: Fix the updatestate for all 3 functions, and join the np2 and np3 all the solvers, a function.
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

    Reduces the number of clauses to the minimum relevant where newvar appears (rest have been already checked)
    @param clauses Array of sets. Each set is a clause of type (int, int int...)
    @param level . How many variable to check. From 1 to level + the new variable
    @param newVariable. the new variable whose valid clause add to the existing
    @param onlynewVariable. False includes all level variables too between themselves, besides newVariable only clauses
    Requires level < newVariable.
"""
def minimClause(clauses, level, newVariable, onlyNewVariable = True):
    varsFilter = set(range(1,level+1))
    varsFilter.add(newVariable)
    minClauses = set()
    for clause in clauses:
        isMinimum = True
        hasNewVariable = False
        for item in clause:
            if abs(item) not in varsFilter:
                isMinimum = False
                break
            elif abs(item) == newVariable:
                hasNewVariable = True
        if isMinimum and ((not onlyNewVariable) or hasNewVariable):
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
-1- delete. passes neither 0 nor 1 newvar instance for this id instance
1- 01b passes newvar 0 
2- 10b passes newvar 1
3- 11b passes newvar 0 and 1
Accpetance condition level < newVariable. Otherwise unexpected behaviour
If numVar is not 0, checks all the variables one per time till numVar
Since newVariable is inside the range already, only is taken into account when numVar = 0
"""

def updateState(clauses, idValue, initStatus, level, newVariable = 0, numVar = 0):
       
    dSet = set()
    for i in range(0,level):
        if (idValue >> i) % 2 == 1:
            idSet.add(i+1)
        else:
            idSet.add(-(i+1))
    #We only loop till newVariable or all the variables.
    if numVar == 0:
        checkStopVar = newVariable
    else:
        checkStopVar = numVar
    finalStatus = 0
    
    for newVariable in range(checkStopVar,level,-1):
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
            elif newVariable != level+1 and numVar != 0:
                continue
  
        #Check if 1 for newVariable was passing. Status 2 or 3
        if (status>>1) % 2 == 1:
            idSet.add(newVariable)
            passes1 = checkBoolean(clauses, instanceSet = idSet)
            idSet.remove(newVariable)
            if not passes1:
                #newVariable at 1 does not pass. Status then 1 or 0
                status = status % 2          
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


def solveCNFGraph(path="", startLevel=10):
    
    # 20 Vars - uf20-01.cnf # 4726 sol. 7714 nodes
    # 50 Vars - uf50-05.cnf
    
    if path == "":
        path = path

    expandids = udf(expandLevel, ArrayType(LongType()))

    graphDF = sc.parallelize([])
    (numVar, numClauses, clauses) = parseNPFunction(path)     
    
    initData = [i for i in range(0,2**(startLevel))]
    initRDD = sc.parallelize(initData)
    initDF = initRDD.map(lambda x: Row(id=long(x), level=startLevel))
    
    graphDF = sqlContext.createDataFrame(initDF)

    
    
    for newvar in range(startLevel+1, numVar+1):
        walkerDF = graphDF.select("id").filter(col("level") == startLevel).withColumn("state",lit(3))
        
        for walklevel in range(startLevel, newvar):
            if newvar == startLevel+1:
                reducedClause = minimClause(clauses, walklevel, newvar,onlyNewVariable = False)
            else:
                reducedClause = minimClause(clauses, walklevel, newvar)
        
            #Parse state with clauses
            walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateState(reducedClause, row['id'], row['state'], walklevel, newvar))).toDF()
            #Creates hash = id mod 2^level in graphDF to delete all branches of a no solution
            graphDF = graphDF.select("*").withColumn("hashID", graphDF['id']%(2**walklevel))
            #joins graphDF with updted status walkerDF and deletes all non solutions at this level, with new variable (marked by state=-1). Puts state=0 (not used) to those ids not existing in walkingDF
            hashGraph = graphDF.join(walkerDF,walkerDF["id"] == graphDF["hashID"],how='leftouter').select(graphDF["id"],col("level"),col("hashID"),col("state")).fillna(0).where(col("state") != -1)
            #Not leaf of tree/graph
            if walklevel+1 != newvar:
                walkerDF = hashGraph.select("id","state").where((hashGraph["state"] != 0) & (col("level") == (walklevel+1)))
                graphDF = hashGraph.select("id", "level")
            #Leaf of tree/graph
            else:
                #Expand new level with the new Variable
                newLevelDF = walkerDF.withColumn("id",explode(array(expandids(col("id"),col("state"),lit(newvar))))).withColumn("level",lit(newvar)).select(explode("id").alias("id"),"level")
                graphDF = graphDF.select("id","level").union(newLevelDF).cache()
    
    return graphDF

"""
solverCNFOnePass
NP Solver that updates a dataset by expanding its solutions when adding a new variable
and filters the ones that do not resolve the clauses for the instanciated variables.
It works on the same dataset of solution by expanding and filtering, one variable each time. In one pass.
"""
def solveCNFOnePass(path="", startLevel=10):
    
    # 20 Vars - uf20-01.cnf # 4726 sol. 7714 nodes
    # 50 Vars - uf50-05.cnf
    
    if path == "":
        path = path

    expandids = udf(expandLevel, ArrayType(LongType()))

    graphDF = sc.parallelize([])
    (numVar, numClauses, clauses) = parseNPFunction(path)     
    
    initData = [i for i in range(0,2**(startLevel))]
    initRDD = sc.parallelize(initData)
    initDF = initRDD.map(lambda x: Row(id=long(x), state=3))
    
    walkerDF = sqlContext.createDataFrame(initDF).cache()
    #graphDF = walkerDF.select("id").withColumn("level",lit(startLevel))

    for newVar in range(startLevel+1, numVar+1):
        if newVar == startLevel+1:
            reducedClause = minimClause(clauses, newVar-1, newVar, onlynewVariable = False)
        else:
            reducedClause = minimClause(clauses, newVar-1, newVar)
        
        #Parse state with clauses
        walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateState(reducedClause, row['id'], row['state'], newVar-1, newVar))).toDF().where(col('state') != -1)
        walkerDF = walkerDF.withColumn("id",explode(array(expandids(col("id"),col("state"),lit(newVar))))).withColumn("state",lit(3)).select(explode("id").alias("id"),"state")
        #graphDF = graphDF.union(walkerDF.select("id").withColumn(lit(newVar)))
        if newVar == 26:
            break;
    return walkerDF.select("id")

def solveCNFResolveAll(path="", startLevel=10):
    
    # 20 Vars - uf20-01.cnf # 4726 sol. 7714 nodes
    # 50 Vars - uf50-05.cnf
    
    if path == "":
        path = path

    expandids = udf(expandLevel, ArrayType(LongType()))

    graphDF = sc.parallelize([])
    (numVar, numClauses, clauses) = parseNPFunction(path)     
    
    initData = [i for i in range(0,2**(startLevel))]
    initRDD = sc.parallelize(initData)
    initDF = initRDD.map(lambda x: Row(id=long(x), state=3))
    
    walkerDF = sqlContext.createDataFrame(initDF).cache()
    #graphDF = walkerDF.select("id").withColumn("level",lit(startLevel))

    for newVar in range(startLevel+1, numVar+1):
        if newVar == startLevel+1:
            reducedClause = minimClause(clauses, newVar-1, newVar, onlynewVariable = False)
        else:
            reducedClause = minimClause(clauses, newVar-1, newVar)
        
        #Parse state with clauses
        walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateState(clauses = clauses, idValue = row['id'], initState = 3, level = newVar-1, numVar = numVar))).toDF().where(col('state') != -1)
        walkerDF = walkerDF.withColumn("id",explode(array(expandids(col("id"),col("state"),lit(newVar))))).withColumn("state",lit(3)).select(explode("id").alias("id"),"state")
        #graphDF = graphDF.union(walkerDF.select("id").withColumn(lit(newVar)))
        if newVar == 30:
            break;
    return walkerDF.select("id")


conf = SparkConf().setAppName("Boolean Solver")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
graphDF = solveCNF("uf20-01.cnf", 10)
graphDF.groupBy("level").count().orderBy("level").show()