# Project Specifications
## Terminology
1. The "perfect" player (PB&J) is given a random "golden" alpha value
2. The "training" player (Peter) is given a beta value
3. Learning rate TBD 

## Axioms
1. Peter is always the first player

## Process
1. After each N iterations where max(N) is TBD, calculate the Mean Squared Error (MSE)
2. Based on the MSE, shift beta by the learning rate in a certain direction
3. Keep iterating 1,2 until MSE ~stops shifting
4. Compare beta to alpha; compare EV on M simulations where M is TBD to -1/18

## Operator Instructions
1. **ALWAYS PULL THE LATEST VERSION OF THE REPO BEFORE RUNNING**
2. Once you finish running however many simulations, if MSE has stopped, see 3. If MSE has not stopped, see 4
3. If MSE has stopped, log the total number of iterations taken in log.md along with the beta/alpha and EV/(-1/18)
4. If MSE has not stopped, log the current total number of iterations taken and current beta value in log.md
5. **MAKE SURE TO PUSH YOUR LOG.MD and BETA VALUE ONTO THE REPO**

## TO DO:
1. Determine N, M, learning rate (lr)
2. Write code for program (optional)
3. Run iterations (4090 preferred!)
**99999**. 3 man Kuhn Poker
