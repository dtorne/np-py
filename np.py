#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:03:33 2017

@author: david
"""

from pyspark.sql import Row
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql.functions import array, explode, lit, col, udf




class NPCMemSolver:
    
    """ 
        Initializes NPCMemSolver with path and the number of variables to expand first fully
    """
    def __init__(self, path, initLevel = 10):
        self._path_ = path
        self._startLevel_ = initLevel
        self._graphDF_ = sc.parallelize([])


    def getStartLevel():
        return self._startLevel_
    
    def getGraphDF():
        return self._graphDF_
    
    
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
      Requires level < newVariable.
    """
    def minimClause(clauses, level, newVariable, onlyNewVariable=True):
        varsFilter = set(range(1,level+1))
        varsFilter.add(newVariable)
        minClauses = set()
        for clause in clauses:
            isMinimum = True
            hasNewVariable = False
            for item in clause:
                if abs(item) not in varsFilter:
                    isMinimum = False
                elif abs(item) == newVariable:
                    hasNewVariable = True
            if isMinimum and ((not onlyNewVariable) or hasNewVariable):
                minClauses.add(clause)
        return minClauses
    
    
    
    
    """
    Calculates if this instantiation passes the closes.
    Returns status. 
    -1- delete. passes neither 0 nor 1 newvar instance for this id instance
    1- 01b passes newvar 0 
    2- 10b passes newvar 1
    3- 11b passes newvar 0 and 1
    Accpetance condition level < newVariable. Otherwise unexpected behaviour
    """
    def updateStatus(clauses, idValue, status, level, newVariable):
        #Returns True if exist solution, False otherwise
        def checkBoolean(clauses, instanceSet):
            for clause in clauses:
                existInstance = True
                for item in clause:
                    if item not in instanceSet:
                        existInstance = False
                if existInstance:
                    return False
            return True
        
        idSet = set()
        for i in range(0,level):
            if (idValue >> i) % 2 == 1:
                idSet.add(i+1)
            else:
                idSet.add(-(i+1))
        #Check if 0 for newVariable was passing. If status 1 or 3
        if status % 2 == 1:
            idSet.add(-newVariable)
            passes0 = checkBoolean(clauses, instanceSet = idSet)
            if not passes0:
                #newVariable at 0 does not pass. Status then 2 or 0
                status = status - 1
            idSet.remove(-newVariable)
        #Check if 1 for newVariable was passing. Status 2 or 3
        if (status>>1) % 2 == 1:
            idSet.add(newVariable)
            passes1 = checkBoolean(clauses, instanceSet = idSet)
            if not passes1:
                #newVariable at 1 does not pass. Status then 1 or 0
                status = status % 2
            idSet.remove(newVariable)
        if status == 0:
            status = -1
        return status
    

    def expandLevel(idCol,stateCol,newVariable):
        result = []
        #We add newVar = 0
        if stateCol % 2 == 1:
            result.append(idCol)
        if (stateCol>>1) % 2 == 1:
            result.append(2**(newVariable-1)+idCol)
        return result
    
    
    def solveCNF():
        
        # 20 Vars - uf20-01.cnf # 4726 sol. 7714 nodes
        # 50 Vars - uf50-05.cnf

        (numVar, numClauses, clauses) = parseNPFunction(path="")
        
        if path == "":
            path = self._path_
        
        startLevel = getStartLevel()
        initData = [i for i in range(0,2**(startLevel-1))]
        initRDD = sc.parallelize(initData)
        initDF = initRDD.map(lambda x: Row(id=int(x), level=startLevel))
        
        graphDF = sqlContext.createDataFrame(initDF)
        
        expand_ids = udf(expandLevel, ArrayType(IntegerType()))
        
        for new_var in range(startLevel+1, numVar+1):
            walkerDF = graphDF.select("id").filter(col("level") == startLevel).withColumn("state",lit(3))
            
            for walk_level in range(startLevel, new_var):
                if new_var == startLevel+1:
                    reducedClause = minimClause(clauses, walk_level, new_var,onlyNewVariable=False)
                else:
                    reducedClause = minimClause(clauses, walk_level, new_var)
            
                #Parse state with clauses
                walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateStatus(reducedClause, row['id'], row['state'], walk_level, new_var))).toDF()
                #Creates hash = id mod 2^level in graphDF to delete all branches of a no solution
                graphDF = graphDF.select("*").withColumn("hashID", graphDF['id']%(2**walk_level))
                #joins graphDF with updted status walkerDF and deletes all non solutions at this level, with new variable (marked by state=-1)
                hashGraph = graphDF.join(walkerDF,walkerDF["id"] == graphDF["hashID"],how='left_outer').select(graphDF["id"],col("level"),col("hashID"),col("state")).where(col("state") != -1)
                #Not leaf of tree/graph
                if walk_level+1 != new_var:
                    walkerDF = hashGraph.select("id","state").where(hashGraph["state"].isNotNull() & (col("level") == (walk_level+1)))
                #Leaf of tree/graph
                else:
                    #Expand new level with the new Variable
                    newLevelDF = walkerDF.withColumn("id",explode(array(expand_ids(col("id"),col("state"),lit(new_var))))).withColumn("level",lit(new_var)).select(explode("id").alias("id"),"level")
                    graphDF = graphDF.select("id","level").union(newLevelDF).cache()
        
        self._graphDF_ = graphDF
    