function [Generatorinit] = Generator_init(opdata,Syn_Machidx_g)

%% 
%%%  Bus No. xd   xq    xd'  xq'  xd''  xq''   Td0'   Tq0'  Td0''  Tq0''   H   D     SgenB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gendata=...
[
    1   1.000/10   0.690/10  0.310/10  0.310/10   0.25/10  0.25/10   10.20  1.500  0.03  0.05  4.200*10  0   100;
    2   2.950/10   2.820/10  0.697/10  0.697/10   0.25/10  0.25/10   6.560  1.500  0.03  0.05  3.030*10  0   100;
    3   2.495/10   2.370/10  0.531/10  0.531/10   0.25/10  0.25/10   5.700  1.500  0.03  0.05  3.580*10  0   100;
    4   2.620/10   2.580/10  0.436/10  0.436/10   0.25/10  0.25/10   5.690  1.500  0.03  0.05  2.860*10  0   100;
    5   6.700/10   6.200/10  1.320/10  1.320/10   0.25/10  0.25/10   5.400  0.440  0.03  0.05  2.600*10  0   100;
    6   2.540/10   2.410/10  0.500/10  0.500/10   0.25/10  0.25/10   7.300  0.400  0.03  0.05  3.480*10  0   100;
    7   2.950/10   2.920/10  0.490/10  0.490/10   0.25/10  0.25/10   5.660  1.500  0.03  0.05  2.640*10  0   100;
    8   2.900/10   2.800/10  0.570/10  0.570/10   0.25/10  0.25/10   6.700  0.410  0.03  0.05  2.430*10  0   100;
    9   2.106/10   2.050/10  0.570/10  0.570/10   0.25/10  0.25/10   4.790  1.960  0.03  0.05  3.450*10  0   100;
   10   0.200/10   0.190/10  0.060/10  0.060/10   0.25/10  0.25/10   7.000  0.700  0.03  0.05  50.00*10  0   100; 
];

Xd=gendata(:,2);
Xq=gendata(:,3);
Xd1=gendata(:,4);
Xq1=gendata(:,5);
Xd11=gendata(:,6);
Xq11=gendata(:,7);
Td0=gendata(:,8);
Tq0=gendata(:,9);
Td10=gendata(:,10);
Tq10=gendata(:,11);
Tj=gendata(:,12);
D=gendata(:,13);
SgenB=gendata(:,14);

Generatorinit.Xd=Xd;
Generatorinit.Xq=Xq;
Generatorinit.Xd1=Xd1;
Generatorinit.Xq1=Xq1;
Generatorinit.Xd11=Xd11;
Generatorinit.Xq11=Xq11;
Generatorinit.Td0=Td0;
Generatorinit.Tq0=Tq0;
Generatorinit.Td10=Td10;
Generatorinit.Tq10=Tq10;
Generatorinit.Tj=Tj;
Generatorinit.D=D;
Generatorinit.SgenB=SgenB;
%% 
% a       b  Tg  dPmax dPmin
%0.00015 0.17 0.25 7.0 -7.0
%  Enable  R      Ts       Tc    T3     T4     T5     dPmax     dPmin   
Govnerdata=...
[
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0
    1     0.00015   0.17    0.25      7.0      -7.0%10
];

Generatorinit.Enablegov  =Govnerdata(:,1);
Generatorinit.a_G        =Govnerdata(:,2).*SgenB./100;
Generatorinit.b_G        =Govnerdata(:,3);
Generatorinit.Tg_G       =Govnerdata(:,4);
Generatorinit.dPmax      =Govnerdata(:,5);
Generatorinit.dPmin      =Govnerdata(:,6);

%% 

Excitdata=...
[  
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0%10

];


Ka=Excitdata(:,1);
Tb=Excitdata(:,2);
Tc=Excitdata(:,3);
Generatorinit.Ka=Ka;
Generatorinit.Tb=Tb;
Generatorinit.Tc=Tc;
Generatorinit.Efmax=Excitdata(:,4);
Generatorinit.Efmin=Excitdata(:,5);

%% 
%%%%%%%%%%%%%%%%%%%15 10 0.07 0.03 0.07 0.03 0.1 -0.1
pssdata=...
[
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;
    1   15    10   0.07     0.03    0.07    0.03     0.1    -0.1;%10  
];
Generatorinit.EnablePSS=pssdata(:,1);
Generatorinit.kpss=pssdata(:,2);
Generatorinit.Tw_P=pssdata(:,3);
Generatorinit.T1_P=pssdata(:,4);
Generatorinit.T2_P=pssdata(:,5);
Generatorinit.T3_P=pssdata(:,6);
Generatorinit.T4_P=pssdata(:,7);
Generatorinit.Upmax=pssdata(:,8);
Generatorinit.Upmin=pssdata(:,9);
%% Initial value calculation 

Pg   =opdata(Syn_Machidx_g,2);
Qg   =opdata(Syn_Machidx_g,3);
Vg   =opdata(Syn_Machidx_g,4);
Theta=opdata(Syn_Machidx_g,5)*pi/180;

Ug=Vg.*exp(1j*Theta);
Sg=Pg+1j*Qg;
Ig=conj(Sg./Ug);

EQ=Ug+1j*Xq.*Ig;
Delta=angle(EQ);
DEL=Delta*180/pi;
Idq=Ig.*(sin(Delta)+1j*cos(Delta));
Udq=Ug.*(sin(Delta)+1j*cos(Delta));

Ef=imag(Udq)+Xd.*real(Idq);
Eq1=imag(Udq)+Xd1.*real(Idq);
Ed1=real(Udq)-Xq1.*imag(Idq);
Eq11=imag(Udq)+Xd11.*real(Idq);
Ed11=real(Udq)-Xq11.*imag(Idq);

Vref=abs(Vg)+Ef./Ka;
Vf1=Ef.*Tb;

%
Generatorinit.Pm0=Pg;
Generatorinit.Delta=Delta;
%
Generatorinit.Eq1=Eq1;
Generatorinit.Eq11=Eq11;
Generatorinit.Eq1=Eq1;
Generatorinit.Eq11=Eq11;
Generatorinit.Ed1  = Ed1;
Generatorinit.Ed11 = Ed11;
%
Generatorinit.Vref = Vref;
Generatorinit.Vf1  = Vf1;



end
