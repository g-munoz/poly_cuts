#!/usr/bin/python

from gurobipy import *
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import math
import sys
import os
import cutgenerators as cuts
import linalgtools as myla
import formulationHandler as fh
import time

EYM = 0
ShiftedEYM = 0
PSD = 1
Minor = 1
wRLT = 0

boundTightening = 1

def main():
	
	print ("Input file: %s" % sys.argv[1] )
	
	## We read a problem from a file. We assume it's quadratic so Gurobi can read it
	m = read(sys.argv[1])
	varlist = m.getVars()
	varNameToNumber = {}
	varNumberToName = {}
	for i in xrange(m.numVars):
		varNumberToName[i] = varlist[i].getAttr("VarName")
		varNameToNumber[varlist[i].getAttr("VarName")] = i
			
	if boundTightening:
		boundImproved = False
		x, X, bound_model = fh.linearize(m, varNameToNumber, varNumberToName, 0)
		
		varlist = m.getVars()
		for i in xrange(m.numVars):
			obj = LinExpr ( x[i] )
			bound_model.setObjective(obj, GRB.MAXIMIZE)
			bound_model.update()
			bound_model.setParam("OutputFlag",0)
			bound_model.optimize()
			
			if bound_model.Status == GRB.OPTIMAL:
				newUB = bound_model.objVal
				if newUB < x[i].getAttr("UB"):
					boundImproved = True
					print 'Upper Bound for x', i, 'changed from', x[i].getAttr("UB"), 'to', newUB
					varlist[i].setAttr("UB", newUB)
				
			bound_model.setObjective(obj, GRB.MINIMIZE)
			bound_model.update()
			bound_model.optimize()
			
			if bound_model.Status == GRB.OPTIMAL:
				newLB = bound_model.objVal
				if newLB > x[i].getAttr("LB"):
					boundImproved = True
					print 'Lower Bound for x', i, 'changed from', x[i].getAttr("LB"), 'to', newLB
					varlist[i].setAttr("LB", newLB)
		if not boundImproved:
			print 'No bound improved'
	
	m.update()
	filename, file_extension = os.path.splitext(sys.argv[1])
	
	## Here we linearize all monomials in quadratics
	x, X, mlin = fh.linearize(m, varNameToNumber, varNumberToName, wRLT)
	Xtovec, vectoX = fh.createMap(x,X)
	
	fullx = [None for i in range(mlin.numVars) ]
	
	for i in xrange(len(x)) :
		fullx[i] = x[i]
	for l in xrange(len(x)):
		for k in xrange(l+1):
			fullx[Xtovec[k][l]] = X[k][l]
	
	c, A, b = fh.buildAb(mlin, x, X, Xtovec)
	
	mlin.setParam("OutputFlag",0)
	mlin.update()
	mlin.optimize()
		
	gurobi_tol = 1E-6

	if not mlin.getAttr("Status") == 2:
		print 'Error, model could not be solved. Gurobi Status', mlin.getAttr("Status")
		sys.exit(2)

	iter_num = 0
	old_val = mlin.objval
	new_val = mlin.objval
	
	iter_stall = 0
	max_iter_stall = 10
	max_iter = 1000000  

	counts = np.zeros(6, dtype=np.int)
	#Count for each type of cut, with:
	#Type 0 = EYM
	#Type 1 = ShiftedEYM
	#Type 2 = PSD
	#Type 3 = Minor
	
	start_time = time.time()
	max_run_time = 3600
	
	gurobi_time = mlin.getAttr("Runtime")
	cut_time = 0
	pre_time = 0
	post_time = 0
	
	rowNorms = [np.linalg.norm(A[i]) for i in xrange(np.shape(A)[0])]
	lenOfFullX = np.shape(A)[1]
	
	print 'Linearized model has', mlin.numVars, 'variables'
	print '==========================================='
	print '{:5s} {:15s} {:15s} {:6s} {:6s}'.format('Iter', 'Max Violation', 'Objective', 'Cuts', 'Time')
	print '==========================================='
	while iter_num < max_iter :
		objval = mlin.objval
		start_t = time.time() 
		
		Abasic, bbasic = computeBasis(mlin, A, b, x, X, Xtovec)
		xbasic = np.linalg.solve(Abasic, bbasic)
		#xgurobi = [fullx[i].X for i in range(len(fullx))]

		lenOfX = len(x)

		xbasic_matrix = cuts.buildMatrixSol(xbasic, Xtovec, lenOfX) #This matrix will be [[1 x][x X]]

		if np.linalg.matrix_rank(xbasic_matrix) == 1:
			print 'Current solution of rank 1, no cut computed', objval
			break
		
		dirs_matrix , dirs = cuts.findraysVecAndMatrix(Abasic, Xtovec, lenOfX, lenOfFullX)

		violation1 = -1
		violation2 = -1
		pi_all = []
		pirhs_all = []
		violation_all = []
		
		pre_time += ( - start_t + time.time() )
		
		start_t = time.time() 

		if EYM:
			pi1, pirhs1, violation1 = cuts.eymcut(xbasic, xbasic_matrix, dirs, dirs_matrix, lenOfX, Xtovec, Abasic, bbasic)
		
			if len(pi1) > 0 :
				pi_all = np.array([pi1])
				pirhs_all = np.array([pirhs1])
				violation_all = np.array([violation1])
				types = np.array([0])
	
		if ShiftedEYM:
			pi1, pirhs1, violation1 = cuts.eymcutplusplus(xbasic, xbasic_matrix, dirs, dirs_matrix, lenOfX, Xtovec, A, b, Abasic, bbasic)

			if len(pi1) > 0:
				if len(pi_all) == 0:
					pi_all = np.array([pi1])
					pirhs_all = np.array([pirhs1])
					violation_all = np.array([violation1])
					types = np.array([1])
				else:
					pi_all = np.vstack([pi_all, pi1])
					pirhs_all = np.hstack([pirhs_all, pirhs1])
					violation_all = np.vstack([violation_all, violation1])
					types = np.hstack([types, 1 ]) 

		if PSD:
			pi_list, pirhs_list, viol_list = cuts.outerpsd(xbasic, xbasic_matrix, lenOfX, Xtovec)

			if not len(pi_list) == 0:
				if len(pi_all) == 0:
					pi_all = pi_list
					pirhs_all = pirhs_list
					violation_all = viol_list
					types = [2 for i in range(len(pirhs_list))] 
				else:	
					pi_all = np.vstack([pi_all, pi_list])
					pirhs_all = np.hstack([pirhs_all, pirhs_list])
					#print np.shape(violation_all), np.shape(viol_list)
					violation_all = np.vstack([violation_all, viol_list])
					types = np.hstack([types, [2 for i in range(len(pirhs_list))] ])

		if Minor:
			pi_list, pirhs_list, viol_list = cuts.minorcut(xbasic, xbasic_matrix, dirs,dirs_matrix, lenOfX, Xtovec, Abasic, bbasic)

			if not len(pi_list) == 0:
				if len(pi_all) == 0:
					pi_all = pi_list
					pirhs_all = pirhs_list
					violation_all = viol_list
					types = [3 for i in range(len(pirhs_list))]
				else:
					pi_all = np.vstack([pi_all, pi_list])
					pirhs_all = np.hstack([pirhs_all, pirhs_list])
					violation_all = np.vstack([violation_all, viol_list])
					types = np.hstack([types, [3 for i in range(len(pirhs_list))] ])
		
		cut_time += ( - start_t + time.time() )
		
		start_t = time.time() 
		
		if len(violation_all) == 0 or max(violation_all) < gurobi_tol:
			print 'Algorithm finished because current pool of cuts does not cut enough'
			break

		# Up to here all cuts are in pi_all, pirhs_all. So now we add them to the model
		skipped = 0
		
		## Here I'll sort the cuts accoring to violation
		cut_tuples = []
		for k in xrange(np.shape(pi_all)[0]):
			if violation_all[k] > 0:
				cut_tuples.append((pi_all[k], pirhs_all[k], violation_all[k], types[k]))
		max_cuts = 5
		added_cuts = 0
		
		cut_tuples.sort(key=lambda curr_tuple: curr_tuple[2], reverse=True)
		
		for k in xrange(len(cut_tuples)):
			
			curr_pi = cut_tuples[k][0]
			curr_pirhs = cut_tuples[k][1]
			curr_violation = cut_tuples[k][2]
			curr_type = cut_tuples[k][3]
			
			if(added_cuts >= max_cuts):
				break
			
			if curr_violation < gurobi_tol:
				continue
						
			if cuts.checkifparallel(curr_pi, curr_pirhs, A, b, rowNorms) == 1:
				skipped += 1
				continue

			added_cuts += 1
			
			A = np.vstack([A, curr_pi])
			b = np.hstack([b, curr_pirhs])
			rowNorms.append(np.linalg.norm(curr_pi))

			cut = LinExpr(np.dot(curr_pi, fullx))
			mlin.addConstr(cut <= curr_pirhs, name = "Type%d_%d" % (curr_type ,counts[curr_type] ))

			counts[curr_type] += 1
		
		end_time = time.time()
		run_time = (- start_time + end_time)
		
		post_time += ( - start_t + time.time() )
		
		if iter_num%5 == 0:
			max_viol = 0
			if len(violation_all) == 1:
				max_viol = violation_all[0]
			else:
				max_viol = max(violation_all)[0]
			if isinstance(max_viol, np.ndarray):
				max_viol = max_viol[0]	
			print '{:5d} {:2.12f} {:2.12f} {:6d} {:3.2f} \t {:6d} parallel cuts skipped'.format(iter_num, max_viol , objval, sum(counts), run_time, skipped)
			
		mlin.update()
		mlin.optimize()
		
		gurobi_time += mlin.getAttr("Runtime")

		if not mlin.getAttr("Status") == 2:
			print 'Error: model could not be solved. Gurobi Status', mlin.getAttr("Status")
			sys.exit(2)

		iter_num += 1
		new_val = mlin.objval
		if abs(new_val - old_val) < eps:
			iter_stall += 1
		else:
			iter_stall = 0
			if new_val - old_val < -eps and mlin.ModelSense > 0:
				print 'Warning!! Numerical instability, finishing algorithm'
				break                
			elif new_val - old_val > eps and mlin.ModelSense < 0:
				print 'Warning!! Numerical instability, finishing algorithm'
				break                

		if iter_stall > max_iter_stall:
			print 'Algorithm finished because objective value did not increase in the last %d iterations' % (max_iter_stall)
			break
		
		if run_time > max_run_time:
			print 'Max runtime reached'
			break
		
		old_val = new_val

	# EYM, SEYM, PSD, Minor
	print '\nINFO1:', counts[0], counts[1], counts[2], counts[3]
	print 'INFO2: {:5d} {:2.12f} {:6d} {:3.2f} {:3.2f} {:3.2f} {:3.2f} {:3.2f}'.format(iter_num, objval, sum(counts), run_time, gurobi_time, pre_time, cut_time, post_time)

	displayBFSSol(mlin, A,b,x,X, Xtovec)

def computeBasis(mlin, A, b, x, X, Xtovec):
	constrs = mlin.getConstrs()

	Abasic = [A[j] for j in range(np.shape(A)[0]) if constrs[j].getAttr("CBasis") == -1]
	bbasic = [b[j] for j in range(len(b)) if constrs[j].getAttr("CBasis") == -1]
		
	if np.shape(A)[1] - np.shape(Abasic)[0] > 0 : 
		BoundsBasis = [[0 for j in range(np.shape(A)[1])] for i in range(np.shape(A)[1] - np.shape(Abasic)[0]) ]
		BoundsRHS = [0 for i in range(np.shape(A)[1] - np.shape(Abasic)[0]) ]
		count = 0
		missing = np.shape(A)[1] - np.shape(Abasic)[0]
		
		for i in xrange(len(x)):
			if x[i].getAttr("VBasis") == -1:
				BoundsBasis[count][i] = -1
				BoundsRHS[count] = -x[i].getAttr("LB")
				count += 1
			elif x[i].getAttr("VBasis") == -2:
				BoundsBasis[count][i] = 1
				BoundsRHS[count] = x[i].getAttr("UB")
				count += 1
				
			if count == missing:
				break
			
		for l in xrange(len(x)):
			for k in xrange(l+1):
				if X[k][l].getAttr("VBasis") == -1:
					BoundsBasis[count][Xtovec[k][l]] = -1
					BoundsRHS[count] = -X[k][l].getAttr("LB")
					count += 1
						
				elif X[k][l].getAttr("VBasis") == -2:
					BoundsBasis[count][Xtovec[k][l]] = 1
					BoundsRHS[count] = X[k][l].getAttr("UB")
					count += 1
				
				if count == missing:
					break
				
			if count == missing:
				break
						
		Abasic = np.vstack([Abasic, BoundsBasis])
		bbasic = np.hstack([bbasic, BoundsRHS])
		
	return Abasic, bbasic


def displayBFSSol(mlin, A,b, x,X, Xtovec):

	Abasic, bbasic = computeBasis(mlin, A, b, x, X, Xtovec)

	if np.shape(Abasic)[0] != np.shape(Abasic)[1]:
		print 'Error with basis given by Gurobi'

	xbasic = np.linalg.solve(Abasic, bbasic)

	xbasic_matrix = cuts.buildMatrixSol(xbasic, Xtovec, len(x)) #This matrix will be [[1 x][x X]]
	
	print '\n======================================'
	print '          SOLUTION SUMMARY            '
	print '======================================'
	print 'The solution matrix [[1 x][x X]] is'
	print('\n'.join(['  '.join(['{:2.7f}'.format(item) for item in row]) for row in xbasic_matrix]))
	eigvals, V = np.linalg.eigh(xbasic_matrix)

	print '\nAccording to numpy its rank is', np.linalg.matrix_rank(xbasic_matrix),'with eigenvalues', eigvals
	print '\nThe closest rank-1 solution from EYM theorem is'
	A = eigvals[len(eigvals)-1]*np.outer(V[:,len(eigvals)-1], V[:,len(eigvals)-1])
	print('\n'.join(['  '.join(['{:2.7f}'.format(item) for item in row]) for row in A]))
	print '\nWhich is the outerproduct of', np.sqrt(eigvals[len(eigvals)-1])*V[:,len(eigvals)-1]*np.sign(V[0,len(eigvals)-1]), 'with itself\n'


if __name__ == "__main__":
	global eps
	eps = 1E-7
	
	if len(sys.argv) >= 3:
		with open(sys.argv[2]) as f:
				for line in f:
					lineVal = line.split()
					if lineVal[0] == 'EYM':
						EYM = int(lineVal[1])
					elif lineVal[0] == 'SEYM':
						ShiftedEYM = int(lineVal[1])
					elif lineVal[0] == 'PSD':
						PSD = int(lineVal[1])
					elif lineVal[0] == 'Minor':
						Minor = int(lineVal[1])
	if len(sys.argv) >= 4:
		wRLT = int(sys.argv[3])
	if len(sys.argv) >= 5:
		boundTightening = int(sys.argv[4])
	
	main()

