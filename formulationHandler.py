#!/usr/bin/python

from gurobipy import *
import numpy as np
import math
import sys

def linearize(m, varNameToNumber, varNumberToName, wRLT):
	## Linearize all quadratic monomials and return new model
	## We assume all variables are named xi
	
	## wRLT determines if weak RLT is used or not.
	
	mlin = m.copy()
	x = mlin.getVars() #save the original variables
	numQConstrs = mlin.numQConstrs
	
	X = [[0 for i in range(mlin.numVars)] for i in range(mlin.numVars)] #save the linearized monomials
	obj = mlin.getObjective()

	if isinstance(obj, gurobipy.QuadExpr) : #if the objective is non-linear

		linear_obj = obj.getLinExpr()

		for j in xrange(obj.size()):
			
			name1 = obj.getVar1(j).getAttr("VarName");
			name2 = obj.getVar2(j).getAttr("VarName");
			
			var1 = varNameToNumber[name1]
			var2 = varNameToNumber[name2]

			if var1 == var2:
				if not isinstance(X[var1][var2], gurobipy.Var): #if not added already
					X[var1][var2] = mlin.addVar(vtype=GRB.CONTINUOUS, name='X(%s,%s)' % (name1,name2), lb = 0)
					mlin.update()

				linear_obj.add(X[var1][var2], obj.getCoeff(j))

			elif var1 < var2:
				if not isinstance(X[var1][var2], gurobipy.Var): #if not added already
					X[var1][var2] = mlin.addVar(vtype=GRB.CONTINUOUS, name='X(%s,%s)' % (name1,name2), lb = -GRB.INFINITY)
					mlin.update()

				linear_obj.add(X[var1][var2], obj.getCoeff(j))

			else:
				if not isinstance(X[var2][var1], gurobipy.Var):
					X[var2][var1] = mlin.addVar(vtype=GRB.CONTINUOUS, name='X(%s,%s)' % (name2,name1), lb = -GRB.INFINITY)
					mlin.update()

				linear_obj.add(X[var2][var1], obj.getCoeff(j))
		mlin.update()
		mlin.setObjective(linear_obj)

	for i in xrange(numQConstrs): #linearize quadratic constraints
		
		q = mlin.getQConstrs()[0]		
		quad = mlin.getQCRow(q)
		linear = quad.getLinExpr()
		
		for j in xrange(quad.size()):
			
			name1 = quad.getVar1(j).getAttr("VarName");
			name2 = quad.getVar2(j).getAttr("VarName");
			
			var1 = varNameToNumber[name1]
			var2 = varNameToNumber[name2]

			if var1 == var2:
				if not isinstance(X[var1][var2], gurobipy.Var): #if not added already
					X[var1][var2] = mlin.addVar(vtype=GRB.CONTINUOUS, name='X(%s,%s)' % (name1,name2), lb = 0)
					mlin.update()

				linear.add(X[var1][var2], quad.getCoeff(j))

			elif var1 < var2:
				if not isinstance(X[var1][var2], gurobipy.Var): #if not added already
					X[var1][var2] = mlin.addVar(vtype=GRB.CONTINUOUS, name='X(%s,%s)' % (name1,name2), lb = -GRB.INFINITY)
					mlin.update()

				linear.add(X[var1][var2], quad.getCoeff(j))

			else:
				if not isinstance(X[var2][var1], gurobipy.Var):
					X[var2][var1] = mlin.addVar(vtype=GRB.CONTINUOUS, name='X(%s,%s)' % (name2,name1), lb = -GRB.INFINITY)
					mlin.update()

				linear.add(X[var2][var1], quad.getCoeff(j))
			
		mlin.update()

		mlin.addConstr(linear ,q.getAttr("QCSense"), q.getAttr("QCRHS"), name = '%s_lin' % q.getAttr("QCName"))
		mlin.remove(mlin.getQConstrs()[0])
		mlin.update()

	# Add missing variables
	for j in xrange(len(x)):
		for i in xrange(j+1):
			if not isinstance(X[i][j], gurobipy.Var):
				namei = varNumberToName[i]
				namej = varNumberToName[j]
					
				if i==j:
					X[i][j] = mlin.addVar(vtype=GRB.CONTINUOUS, lb = 0, name='X(%s,%s)' % (namei,namej))
					mlin.update()
				else:
					X[i][j] = mlin.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name='X(%s,%s)' % (namei,namej))

	mlin.update()
	addRLTconstraints(mlin,x,X, wRLT)
	mlin.update()
	return (x,X,mlin)

def addRLTconstraints(m,x,X, wRLT):
        if wRLT:
            print 'Relaxing using wRLT'
        else:
            print 'Relaxing using RLT'

	for j in xrange(len(x)):
		ubj = x[j].getAttr("UB")
		lbj = x[j].getAttr("LB")
		for i in xrange(j+1):
			ubi = x[i].getAttr("UB")
			lbi = x[i].getAttr("LB")
		
			#this is for wRLT
			if wRLT and i == j:                        
				m.addConstr( X[i][j] <= abs(ubi*lbi), name = 'RLT4(%d,%d)' %(i,j) )
				
			elif isinstance(X[i][j], gurobipy.Var):
				if (not lbj == - GRB.INFINITY) and (not lbi == - GRB.INFINITY):
					m.addConstr( X[i][j] - lbj*x[i] - lbi*x[j] + lbi*lbj >= 0, name = 'RLT1(%d,%d)' %(i,j) )
				if (not ubj == GRB.INFINITY) and (not ubi == GRB.INFINITY):
					m.addConstr( X[i][j] - ubj*x[i] - ubi*x[j] + ubi*ubj >= 0, name = 'RLT2(%d,%d)' %(i,j) )
				if (not lbj == - GRB.INFINITY) and (not ubi == GRB.INFINITY):
					m.addConstr( X[i][j] - lbj*x[i] - ubi*x[j] + ubi*lbj <= 0, name = 'RLT3(%d,%d)' %(i,j) )
				if (not ubj == GRB.INFINITY) and (not lbi == - GRB.INFINITY):
					m.addConstr( X[i][j] - ubj*x[i] - lbi*x[j] + lbi*ubj <= 0, name = 'RLT4(%d,%d)' %(i,j) )



def createMap(x,X):
	# Create map between X and its vector form, to go back and forth.
	
	Xtovec = [[None for i in range(len(x))] for i in range(len(x))]
	vectoX = []
	count = 0

	for j in xrange(len(x)):
		for i in xrange(j+1):
			if isinstance(X[i][j], gurobipy.Var):
				Xtovec[i][j] = len(x) + count
				vectoX.append((i,j))
				count += 1

	return (Xtovec, vectoX)


def buildAb(m, x, X, Xtovec):

	# build the matrix of constraints A*y <= b
	# where y = vec([x X])

	A = [[0 for i in range(m.numVars)] for i in range(m.numConstrs)]
	b = [ 0 for i in range(m.numConstrs) ]
	c = [0 for i in range(m.numVars)]
	
	constrs = m.getConstrs()
	
	# Now we go over the constraints and save the coefficients in A
	for j in xrange(m.numConstrs):
		row = m.getRow(constrs[j])
		sense = constrs[j].getAttr("Sense")
		flip = 1
		if sense == '>':
			b[j] = - constrs[j].getAttr("RHS")
			flip = -1
		else:
			b[j] = constrs[j].getAttr("RHS")

		for i in xrange(row.size()):
			coeff = row.getCoeff(i)
			var = row.getVar(i)
			found = 0
			if not var.getAttr("VarName")[0] == 'X':
				for varindex in xrange(len(x)):
					if var.sameAs(x[varindex]):
						A[j][varindex] = flip*coeff
						break
			else:
				for l in xrange(len(x)):
						for k in xrange(l+1):
							if var.sameAs(X[k][l]):
								A[j][ Xtovec[k][l] ] = flip*coeff
								found = 1
								break
						if found == 1:
							break

	obj = m.getObjective()
	for i in xrange(obj.size()):
		coeff = obj.getCoeff(i)
		var = obj.getVar(i)
		found = 0
		if not var.getAttr("VarName")[0] == 'X':
			for varindex in xrange(len(x)):
				if var.sameAs(x[varindex]):
					c[varindex] = coeff
					break
		else:
			for l in xrange(len(x)):
					for k in xrange(l+1):
						if var.sameAs(X[k][l]):
							c[ Xtovec[k][l] ] = coeff
							found = 1
							break
					if found == 1:
						break
	
	return (c,A,b)
