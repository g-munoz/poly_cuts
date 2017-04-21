#!/usr/bin/python

from gurobipy import *
import numpy as np
import math
import sys
import linalgtools as myla
import time

global eps
global normalizeQuadratics

eps = 1E-8
cut_eps = 1E-5
normalizeQuadratics = False
input_opt = 0
strengthen = False

def checkifparallel(pi, pirhs, A, b, rowNorms):
	
	normpi = math.sqrt(np.sum(np.multiply(pi, pi)))

	for i in range(len(b)):
		v = A[i]
		normv = rowNorms[i]
		inner = np.dot(pi,v)
		if normpi*normv - abs(inner) < eps and  inner > 0 and b[i]/normv <= pirhs/normpi :#only allow parallel when stronger
				return 1
	return 0
		
def removenearparallel(A, b, m):
	#print A
	#print m.getConstrs()
	count = 0
	i = 0
	while i < m.numConstrs:
		j = i+1
		while j < m.numConstrs:
			u = A[i]
			v = A[j]
			normu = np.linalg.norm(u)
			normv = np.linalg.norm(v)
			if normu*normv - abs(np.dot(u,v)) < eps and  np.dot(u,v) > 0: 
				#if vectors are near parallel and not in 180 degrees
				if  b[i]/normu < b[j]/normv : #if i is tighter than j, remove j
					m.remove(m.getConstrs()[j])
					m.update()
					A = np.delete(A, j, 0)
					b = np.delete(b, j, 0)
					count += 1
					continue #continue loop without increasing j
				else:
					m.remove(m.getConstrs()[i])
					m.update()
					A = np.delete(A, i, 0)
					b = np.delete(b, i, 0)
					count += 1
					if j > i+1:
						j -= 1
					continue

			else:
				j += 1
		i += 1
	return A,b,count

def minorcut(solX, solX_matrix, dirs, dirs_matrix, n, Xtovec, Abasic, bbasic):
	#generate elementary minor cuts
	#to ensure separation, problem MUST have nonnegative diagonals
	eigtol = -1E-9

	elemvioltol=1E-5; #pminor violation when to add a cut, 10^-6 seems to give stability problems
	stepback=1E-8; #what fraction to step back to avoid infeas cuts
	betadigits=12; #how much to round beta, only for inversion stability
	tolerance=1E-6; #tolerance to send out a warning about not cutting off x due to basis problem
	disctol = -1E-4; #quadratic root formula may have slightly imaginary solutions from numerical error
	ranktol = 1E-9; #determines if the direction submatrix is rank one
	diagtol = 1E-9; #PSD check condition for extreme ray direction (shows containment)
	dettol= 1E-9; #PSD check condition for extreme ray direction (shows containment) for safety, check for epsilon psd i.e. pd
	PDtol=1E-8

	#find elementary violations
	viol_idx = []
	counter=0;
	maxviol=0;
	
	minor_violation_tuples = []
	max_cuts = 10

	for i in xrange(n):
		for j in xrange(i+1, n + 1):

			if (solX_matrix[i][i]*solX_matrix[j][j] - solX_matrix[i][j]**2) > eps :
				
				normTerm = math.sqrt((solX_matrix[i][i]- solX_matrix[j][j])*(solX_matrix[i][i]- solX_matrix[j][j]) + 4*solX_matrix[i][j]*solX_matrix[i][j])
				
				normviol = ( solX_matrix[i][i] + solX_matrix[j][j] - normTerm)/(2*(solX_matrix[i][i] + solX_matrix[j][j])) #min eig/trace
				maxviol = max(maxviol,normviol)
				if normviol > elemvioltol :
					if viol_idx == [] :
						viol_idx = np.array([i, j])
					else:
						viol_idx = np.vstack([viol_idx, [i , j]])
					counter+=1
					
					minor_violation_tuples.append(([i,j], normviol))

	if counter == 1:
		viol_idx = viol_idx.reshape((1,len(viol_idx)))
    	
	if viol_idx == [] :
		#'no pminor violations.'
		return [], [] , []
	
	viol_len = np.shape(viol_idx)[0]	
	
	pi = np.zeros((min(viol_len, max_cuts), len(solX)))
	pirhs = np.zeros(min(viol_len, max_cuts))
	violations = np.zeros(min(viol_len, max_cuts))
	
	#generate elementary cuts
	for vind in xrange(min(viol_len, max_cuts)):
		ind1 = viol_idx[vind][0];
		ind2 = viol_idx[vind][1];

		#find intersection points
		Beta = np.zeros(len(solX))
		
		for k in xrange(len(solX)):
			D = dirs_matrix[k]
			quadA = D[ind1][ind2]**2 - D[ind1][ind1]*D[ind2][ind2]  #determinant of principal minor of directions
			quadB = 2*solX_matrix[ind1][ind2]*D[ind1][ind2] - D[ind1][ind1]*solX_matrix[ind2][ind2] - solX_matrix[ind1][ind1]*D[ind2][ind2]
			quadC = solX_matrix[ind1][ind2]**2 - solX_matrix[ind1][ind1]*solX_matrix[ind2][ind2]

			#normalization for stability?
			#if normalizeQuadratics:
				#max_val = max([abs(quadA), abs(quadB), abs(quadC)])
				#max_val = quadA
				#quadB = quadB/max_val
				#quadC = quadC/max_val
				#quadA = quadA/max_val

			if quadA < dettol and D[ind1][ind1] > diagtol and D[ind2][ind2] > diagtol :
				#PD matrix + PSD matrix is PD, so ex. ray is in cone
				valpos = float("inf")
				valneg = float("inf")
		
			elif abs(quadA) < ranktol :
				#quad formula is approx linear, use more stable lin eq.
				if abs(quadB) < eps:
					valpos = float("inf")
					valneg = float("inf")
				else:
					valpos = quadC/(-quadB)
					valneg = valpos
					
			else:
				
				discflag = 0;
				if quadB**2 - 4*quadA*quadC < 0:
					#since we are in interior of a cone, the discriminant
					#should be nonnegative (i.e. intersection exists in at
					#least one direction)
					if( (quadB**2 - 4*quadA*quadC)/max(quadB**2,abs(4*quadA*quadC )) > disctol): #almost 0, normalized
						discflag=1
					else:
						print 'Discriminant negative, this should not be possible.'
						return None, None, None

				if discflag == 1:
					valpos = (-quadB)/(2*quadA)
					valneg = valpos
				else :
					valpos = (-quadB+math.sqrt(quadB**2-4*quadA*quadC))/(2*quadA)
					valneg = (-quadB-math.sqrt(quadB**2-4*quadA*quadC))/(2*quadA)                    
						
			if max(valpos,valneg) < 0:  #PSD and NSD cone boundaries are behind the point, ray is contained in PSD cone
				steplength = float("inf")
			elif min(valpos,valneg) < 0 :  #PSD cone in front and behind
				steplength = max(valpos,valneg) 
			else:  #hits the PSD cone first, then goes through the NSD cone (tail is unbounded)
				steplength = min(valpos,valneg)
				
			steplength = steplength*(1-stepback)
			if steplength < eps:
				pirhs[vind] = 0
				violations[vind] = 0
				continue
			Beta[k] = 1.0/steplength    #make sure step keeps us in bounds
               
		if np.linalg.norm(Beta) == 0 :
			print 'minor cut shows infeasible problem'
			return [], [], []

		#STRENGTHENING STARTS HERE
		if strengthen:
			if np.linalg.matrix_rank(np.outer(Beta,solX)+ dirs) < len(solX):
				zerocontflag = 1
			else:
				zerocontflag = 0
		    
			if zerocontflag :
				constmat = solX_matrix
			else :
				bdind = np.where(Beta > eps)[0][0] #find the index of any finite steplength row
				constmat = -dirs_matrix[bdind]/Beta[bdind] #apex - (apex + step(bdind)*direction(bdind)) 
	  
			for i in xrange(len(solX)) :
				if abs(Beta[i]) > eps: #Only enter the loop when Beta[i] == 0
					continue
				D = dirs_matrix[i]
				quadA = - D[ind1][ind2]**2 + D[ind1][ind1]*D[ind2][ind2] 

				if quadA > PDtol and D[ind1][ind1] > PDtol and D[ind2][ind2] > PDtol :
				    #tighten contained rays by rotation
				    #let C = constmat, D = dirmat
				    #want C+alpha*D to be exactly psd:
				    #(C11+alphaD11)(C22+alphaD22)=(C12+alphaD12)^2
				    #C11C22+alphaC11D22+alphaC22D11+alpha^2D11D22=C12^2+2alphaC12D12+alpha^2D12^2
				    #a=D11D22-D12^2
				    #b=C11D22+C22D11-2C12D12
				    #c=C11C22-C12^2
				    #ax^2+bx+c, want the nonpositive root closest to zero
					a = D[ind1][ind1]*D[ind2][ind2] - D[ind1][ind2]**2
					b = constmat[ind1][ind1]*D[ind2][ind2] + constmat[ind2][ind2]*D[ind1][ind1] - 2*constmat[ind1][ind2]*D[ind1][ind2]
					c = constmat[ind1][ind1]*constmat[ind2][ind2] - constmat[ind1][ind2]**2
					
					if normalizeQuadratics:
						max_val = max([abs(a), abs(b), abs(c)])
						b = b/max_val
						c = c/max_val
						a = a/max_val
					
					discr = b**2 - 4*a*c
					if discr/max(b**2, abs(4*a*c)) < -eps: #dicriminant normalized
						print 'Error: discriminant should be nonnegative'
						return
					elif discr/max(b**2, abs(4*a*c)) > -eps and discr < 0:
						print 'Warning: discriminant negative but small. Corrected to 0'
						discr = 0
				
					cand1 = (-b + math.sqrt(discr))/(2*a)
					cand2 = (-b - math.sqrt(discr))/(2*a)
					    
					newstep = min(cand1,cand2)
					    
					newstep = newstep*(1-stepback) #negative value, negative stepback is the safe direction
					Beta[i] = 1.0/newstep
					#print 'Direction strengthened'
			 
		#STRENGTHENING ENDS HERE
		
		#Balas' Formula
		pi[vind] = np.dot(Beta, Abasic)
		pirhs[vind] = np.sum(np.multiply(Beta, bbasic)) - 1
		
		flip = (np.sum(np.multiply(pi[vind], solX)) < pirhs[vind])
		
		#(1-2*flip) replaces if. Takes value 1 when flip = 0, and -1 if flip = 1
		pi[vind] = (1-2*flip)*pi[vind]
		pirhs[vind] = (1-2*flip)*pirhs[vind]
		
		norm = sum(abs(pi[vind]))
		
		pi[vind] = pi[vind]/norm
		pirhs[vind] = pirhs[vind]/norm

		violations[vind] = np.sum(np.multiply(pi[vind], solX)) - pirhs[vind]

	violations = np.array(violations)
	violations = violations.reshape((len(violations),1))

	return pi, pirhs, violations 

def outerpsd(solX, solX_matrix, n, Xtovec):

	eigtol = -1E-3
	d, v = np.linalg.eigh(solX_matrix)

	if min(d) > eigtol :
		return [], [], [0, 0]

	pi_list = None
	pirhs_list = None
	viol_list = None
	for i in xrange(n + 1):
		if  d[i] < eigtol:
			#%as long as square numbers the cuts are valid, so round the
			#%vector first
			#rvec=round(v(:,i),rounddigits);
			rvec = v[:,i]
			pi = np.zeros(len(solX))
			pirhs = 0
			for j in xrange(n):
				pi[j] = -2*v[j+1][i]*v[0][i]

				for k in xrange(j+1) :

					if not Xtovec[k][j] == None:
						if not k == j :
							pi[Xtovec[k][j]] = -2*v[k+1][i]*v[j+1][i]
						else :
							pi[Xtovec[k][j]] = -v[k+1][i]**2

					else: #in this case, the bilinear term is exact
						print 'Outer PSD cannot handle non-existent terms'
						return [], [], [0, 0]

			pirhs = v[0][i]**2

			norm = sum(abs(pi))
			
			pi = pi/norm
			pirhs = pirhs/norm
			
			violation = np.dot(pi, solX) - pirhs

			if not pirhs_list == None:
				pi_list = np.vstack([pi_list, pi])
				pirhs_list.append(pirhs)
				viol_list.append(violation)
			else:
				pirhs_list = [0]
				viol_list = [0]

				pi_list = pi.reshape((1,len(pi)))
				pirhs_list[0] = pirhs
				viol_list[0] = violation

	viol_list = np.array(viol_list)
	viol_list = viol_list.reshape((len(viol_list),1))
	return pi_list, pirhs_list, viol_list


def eymcutplusplus(solX, solX_matrix, dirs, dirs_matrix, nx, Xtovec, A, b, Abasic, bbasic):
	
	stepback = 1E-8
	bignum = 1E5

	#given polyhedron defined by Ax<=b, generate EYM
	#intersection ball cut centered at (or near) x
	#return coeffs for the cut pi(x) <= pirhs

	# I only call this if the matrix was not rank 1
	#if np.linalg.matrix_rank(solX_matrix) == 1:
		#print 'Current solution of rank 1, no cut computed'
		#return [], [], []
   
	eigvals, V = np.linalg.eigh(solX_matrix)
	n = len(eigvals)
	
	tempmat = solX_matrix - eigvals[n-1]*np.outer(V[:,n-1],V[:,n-1])
	radius = math.sqrt(np.sum(np.multiply(tempmat, tempmat)))
	
	if eigvals[n-1] > eps :
		if eigvals[n-2] <= eps : #cut is given by opf halfspace, can replace rhs with small pos num
			print 'HALFSPACE'
			tempmat2 = eigvals[n-1]*np.outer(V[:,n-1],V[:,n-1])
			tempmat = 2*tempmat - np.diag(np.diag(tempmat))
            
			pi = buildSolFromMatrix(tempmat, Xtovec, len(solX))
			pirhs = np.sum(np.multiply(tempmat,tempmat2))
            
		else:
			#print 'CONE CUT'
			tempmat = eigvals[n-1]*np.outer(V[:,n-1],V[:,n-1])
			ratio = (eigvals[n-1]/eigvals[n-2])
			C = tempmat + ratio*(solX_matrix - tempmat)
            
			q = ratio*radius
            #Shifted ball has centre C and radius q
            
			Cfro = math.sqrt(np.sum(np.multiply(C,C)))
			Csqs = Cfro**2 - q**2

			innerXC = np.sum(np.multiply(solX_matrix,C) )
			Z3 = Cfro*solX_matrix - innerXC*C/Cfro
            
			Beta = np.zeros(len(solX))
            
			infflags = np.zeros(np.shape(dirs)[0])
			testflag = 1
            
			for i in xrange(np.shape(dirs)[0]):
				D = dirs_matrix[i]
				
                #check for finite step length
				#innerDC2 = np.sum(np.multiply(D,C))
				innerDC = np.sum(np.multiply(D,C))
                
				r1 = innerDC*q/(Cfro*math.sqrt(Csqs))
				
				auxMatrix = D-innerDC*C/(Cfro*Cfro)
				d1 = math.sqrt(np.sum(np.multiply(auxMatrix,auxMatrix))) 
                
				if innerDC < eps or d1 + eps > r1 :
					#finite intersection
					Z4 = Cfro*D - innerDC*C/Cfro
					quada = (q*innerDC)**2/Csqs - np.sum(np.multiply(Z4,Z4));
					quadb = 2*(q**2)*innerXC*innerDC/Csqs - 2*np.sum(np.multiply(Z3,Z4));
					quadc = (q*innerXC)**2/Csqs - np.sum(np.multiply(Z3,Z3));
					
					linear= 0
					
					if abs(quada) < eps and abs(quada/quadb) < eps: 
						steplength = -quadc/quadb
						linear = 1
						
					else:
						
						discriminant = quadb**2 - 4*quada*quadc;
						
						if discriminant < 0 and discriminant/max(quadb**2,abs(4*quada*quadc) ) > -eps:
							discriminant = 0
						elif discriminant/max(quadb**2,abs(4*quada*quadc) ) < -eps:
							print 'Negative discriminant!!!', discriminant
							return [], [], []
					
						sqrtterm = math.sqrt(discriminant)
						plusquad = (-quadb + sqrtterm)/(2*quada);
						minusquad = (-quadb - sqrtterm)/(2*quada);
						
						if min(plusquad,minusquad) > 0 :
							steplength=min(plusquad,minusquad);  
						elif max(plusquad,minusquad) < 0 :
							print 'VERY WEIRD ERROR'
							steplength = radius/np.linalg.norm(D, 'fro') 
						else :
							steplength = max(plusquad,minusquad)
						
					steplength = (1-stepback)*steplength
					
					testpt = solX_matrix + steplength*D
                    
					testr = np.sum(np.multiply(testpt,C))*q/(Cfro*math.sqrt(Csqs))
					
					auxMatrix = testpt-np.sum(np.multiply(testpt,C))*C/(Cfro*Cfro)
					testd = math.sqrt(np.sum(np.multiply(auxMatrix, auxMatrix)))
                    
					if testr < testd :
						print 'STEPLENGTH ERROR, REVERTING ---------'
						steplength = radius/np.linalg.norm(D, 'fro')
						
					elif (testr-testd)/testd > 0.01 :
						print 'big gap?'
						print testr, 'vs', testd
						if radius/np.linalg.norm(D, 'fro') > steplength :
							print 'steplength is smaller than ballstep, steplength:'
							print steplength
							print 'ballstep: '
							print radius/np.linalg.norm(D, 'fro')
							steplength = radius/np.linalg.norm(D, 'fro')
					
					lamDm=steplength*D; #for strengthening
					testflag=0;
					
				else :
					#INTERSECTION AT INFINITY
					steplength = float("inf")
					infflags[i] = 1
					
				Beta[i] = 1/steplength

			#strengthen infinite steps
			innerLC = np.sum(np.multiply(lamDm,C))
			Z1 = Cfro*lamDm - innerLC*C/Cfro
            
			for i in xrange(np.shape(dirs)[0]):
				if infflags[i] :
					D = dirs_matrix[i]
					#D = D/np.linalg.norm(D, 'fro') 

					innerDC = np.sum(np.multiply(D,C))
					Z2 = innerDC*C/Cfro-Cfro*D
					quada = (q*innerDC)**2/Csqs - np.sum(np.multiply(Z2,Z2))
					quadb = -2*(q**2)*innerLC*innerDC/Csqs - 2*np.sum(np.multiply(Z1,Z2))
					quadc = (q*innerLC)**2/Csqs - np.sum(np.multiply(Z1,Z1))
                    
					discriminant = quadb**2 - 4*quada*quadc
                    
					if discriminant < 0 and discriminant/max(quadb**2,abs(4*quada*quadc) ) > -eps:
						discriminant = 0
					elif discriminant/max(quadb**2,abs(4*quada*quadc) ) < -eps:
						print 'Negative discriminant!!!', discriminant
						return [], [], []
                    
					sqrtterm = math.sqrt(discriminant)
					plusquad = (-quadb+sqrtterm)/(2*quada)
					minusquad = (-quadb-sqrtterm)/(2*quada)
                        
					if min(plusquad,minusquad) < 0 :
						steplength = min(plusquad,minusquad)
						steplength = (1+stepback)*steplength #note this has to be plus, you want this to be more negative
						testdir = lamDm - steplength*D
						rtest = np.sum(np.multiply(testdir,C))*q/(Cfro*math.sqrt(Cfro**2 - q**2))
						
						auxMatrix = testdir-np.sum(np.multiply(testdir,C))*C/(Cfro*Cfro)
						dtest = math.sqrt(np.sum(np.multiply(auxMatrix,auxMatrix)))

						if dtest <= rtest:
							Beta[i] = 1/steplength
						else :
							print 'Tilt direction not in recession cone, reverting' #leave step at infinity
                            
					else :
						print 'Tilt error, positive steps, reverting' #leave step at infinity
                        
			pi = np.dot(Beta, Abasic)
			pirhs = np.dot(Beta, bbasic) - 1
			
	else :
		print 'OA CUT'
		pi = solX
		pirhs = 0
    
	flip = np.dot(pi, solX) < pirhs
	pi = (1-2*flip)*pi
	pirhs = (1-2*flip)*pirhs
    
	norm = np.linalg.norm(pi,1)
	pi = pi/norm
	pirhs = pirhs/norm

	violation = np.dot(pi, solX) - pirhs

	return pi, pirhs, violation
   

def shifteymcut(solX, solX_matrix, dirs, dirs_matrix, nx, Xtovec, A, b, Abasic, bbasic):
	stepback = 1E-8
	bignum = 1E4

	#given polyhedron defined by Ax<=b, generate EYM
	#intersection ball cut centered at (or near) x
	#return coeffs for the cut pi(x) <= pirhs
   
	eigvals, V = np.linalg.eigh(solX_matrix)
	n = len(eigvals)
	
	if eigvals[n-2] > eps :
        	shiftfact = (eigvals[n-1] - eigvals[n-2])/eigvals[n-2];
    	else :
    		shiftfact = bignum;

    	matdir = solX_matrix - eigvals[n-1]*np.outer(V[:,n-1],V[:,n-1])
    	shiftX = solX_matrix + matdir*shiftfact

	eigvals, V = np.linalg.eigh(shiftX)
	radius = np.linalg.norm(shiftX - eigvals[n-1]*np.outer(V[:,n-1],V[:,n-1]),'fro') 
	
 	Beta = np.zeros(len(solX));
        tempA = solX_matrix - shiftX
	for k in xrange(np.shape(dirs)[0]):
		D = dirs_matrix[k]

		quadA = np.linalg.norm(D,'fro')**2
		quadB = 2*np.sum(np.multiply(tempA, np.transpose(D)))
		quadC = np.linalg.norm(tempA,'fro')**2 - radius**2

		if normalizeQuadratics:
			max_val = max([abs(quadA), abs(quadB), abs(quadC)])
			quadB = quadB/max_val
			quadC = quadC/max_val
			quadA = quadA/max_val

		disc = quadB**2 - 4*quadA*quadC
		if disc < 0 and disc/max(quadB**2,abs(4*quadA*quadC) ) > -eps:
			disc = 0
		elif disc/max(quadB**2,abs(4*quadA*quadC) ) < -eps:
			print 'Negative discriminant!!!', disc, 'radius', radius
			return [], [], []
			#print eigvals

		steplength = (1-stepback)*(-quadB + math.sqrt(disc))/(2*quadA)
		Beta[k] = 1/steplength
        		
	pi = np.dot(Beta, Abasic)
	pirhs = np.dot(Beta, bbasic) - 1
    
	if np.dot(pi, solX) < pirhs:
        	pi = -pi
        	pirhs = -pirhs
    
	norm = np.linalg.norm(pi,1)
	pi = pi/norm
	pirhs = pirhs/norm

	violation = np.dot(pi, solX) - pirhs

	return pi, pirhs, violation

def eymcut(solX, solX_matrix, dirs, dirs_matrix, n, Xtovec, Abasic, bbasic):

	stepback = 1E-7

	#given polyhedron defined by Ax<=b, generate EYM
	#intersection ball cut centered at (or near) x
	#return coeffs for the cut pi(x) <= pirhs

	[eigvals,vtemp] = np.linalg.eig(solX_matrix);

	if max(eigvals) < 0:
		radius = (1-stepback)*np.linalg.norm(solX_matrix,'fro')
	else:
		eigvals = sorted(eigvals, reverse=True) #
		radius = (1-stepback)*np.linalg.norm(eigvals[1:]);

	Beta = np.zeros(len(solX));
    
	for k in xrange(np.shape(dirs)[0]):
		D = dirs_matrix[k]
		Beta[k]= np.linalg.norm(D,'fro')/radius; ## Inverse step length	

	pi = np.dot(Beta, Abasic)
	pirhs = np.dot(Beta, bbasic) - 1

	flip = np.dot(pi, solX) < pirhs
	pi = (1-2*flip)*pi
	pirhs = (1-2*flip)*pirhs
    
	norm = np.linalg.norm(pi,1)
	pi = pi/norm
	pirhs = pirhs/norm
	
	violation = np.dot(pi, solX) - pirhs

	return pi, pirhs, violation


def findraysVecAndMatrix(Abasic, Xtovec, n, lenofFullX):
	
	dirs = -np.transpose(np.linalg.inv(Abasic))
	dirs_matrix = np.zeros((lenofFullX, 1+n, 1+n ))
	
	for k in xrange(lenofFullX):
		D = dirs_matrix[k]
		dir_vec = dirs[k]
		for j in xrange(n):
			val = dir_vec[j]
			D[j+1][0] = val
			D[0][j+1] = val

			for i in xrange(j+1):
				#if not Xtovec[i][j] == None:
				val = dir_vec[Xtovec[i][j]]
				D[i+1][j+1] = val
				D[j+1][i+1] = val
				
	return dirs_matrix, dirs

def createDistanceMatrix(dir_vec, Xtovec, n):

	D = np.zeros( (1+n, 1+n) )

	for j in xrange(n):
		val = dir_vec[j]
		D[j+1][0] = val
		D[0][j+1] = val

		for i in xrange(j+1):
			#if not Xtovec[i][j] == None:
			val = dir_vec[Xtovec[i][j]]
			D[i+1][j+1] = val
			D[j+1][i+1] = val
	return D

def buildMatrixSol(solX, Xtovec, n): #We assume solX = [x, vect(X)]

	solX_matrix = np.zeros( (1+n, 1+n) )	
	solX_matrix[0][0] = 1

	for j in xrange(n):
		solX_matrix[j+1][0] = solX[j]
		solX_matrix[0][j+1] = solX[j]

		for i in xrange(j+1):

			if not Xtovec[i][j] == None:
				solX_matrix[i+1][j+1] = solX[Xtovec[i][j]]
				solX_matrix[j+1][i+1] = solX[Xtovec[i][j]]
			else: # We fill the non-existent variables with constants
				solX_matrix[i+1][j+1] = solX[j]*solX[i]
				solX_matrix[j+1][i+1] = solX[j]*solX[i]

	return solX_matrix

def buildSolFromMatrix(solX_matrix, Xtovec, n):
	solX = np.zeros(n)
	
	m = np.shape(solX_matrix)[0] - 1
	
	for j in xrange(m):
		solX[j] = solX_matrix[j+1][0]
		
		for i in xrange(j+1):
			solX[Xtovec[i][j]] = solX_matrix[i+1][j+1]
			
	return solX
