# Crowd-Bouncing-Load-Simulation
This project can generate crowd bouncing load time histories of any desired duration and number of bouncing individuals.  
The "ImpulseSimulation" is used to generate a large number of stochastic bouncing impulses. (Python code by Pytorch framework)  
The "PowerSequenceSimulation" and "TimeSequenceSimulation" are used to generate power sequences and time interval sequences with the number of the desired number of bouncing individuals. (Python code by Pytorch framework)  
The "CrowdLoadGneration" contains the script used to generte load time histories by concatenating the generated impulses according to the generated power sequences and the time interval sequences. (Matlab code)  
  
*-----------------------------------------------------------------------------------------------------------------------*  
The necessary packages for the "ImpulseSimulation", "PowerSequenceSimulation", and "TimeSequenceSimulation" are as belows  
  
**PackageNames** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **Version**  
torch(Namely Pytorch) &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; 1.9.1  
numpy &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.24.3  
scipy &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 1.10.1  
matplotlib &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp; 3.7.1  
