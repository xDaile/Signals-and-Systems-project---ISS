import numpy as np
import soundfile as sf
import IPython
from scipy.signal import spectrogram, lfilter, freqz,freqs, tf2zpk
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
print("Import Succesfull")
s, fs= sf.read('xzelen24.wav')

#uloha cislo 1
print("****************ULOHA CISLO 1**************")
print ("vzorkovacia frekvencia:\t",fs)
print("dlzka vo vzorkoch:\t",len(s))
print("dlzka v sekundach:\t", len(s)/fs)
print("počet repr, bin. simbolov:\t",len(s)/16)
print("*************KONIEC ULOHY CISLO 1**********\n\n")
print("****************ULOHA CISLO 2**************")

#koniec ulohy cislo 1
i=0
s=s[:32000]
q=list()
while (i< len(s)-8 ):
    i=i+8
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    if(s[i]>0):
        q.append(1)

    if(s[i]<0):
        q.append(0)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    q.append(None)
    i=i+8
#need to compute 0.2sec of signal
prot3=q
plt.figure(figsize=(15,3))
sig=(len(s)/((len(s)/fs))/50)
s=s[:int(sig)]
forprotoc=np.arange(s.size)/fs

plt.plot(forprotoc,s, 'g',linewidth=1)
q=q[:int(sig)]
plt.scatter(forprotoc,q)
#plt.plot(q, 'r',linewidth=4.1)
#hovadiny okolo grafu

plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('sig[value]')
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
#koniec hovadin
#DELETE plt.show()

s, fs= sf.read('xzelen24.wav')
t=np.arange(s.size)/fs
plt.figure(figsize=(15,3))
plt.plot(t,s)
#hovadiny okolo grafu
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
#koniec hovadin
#DELETE plt.show()
#print (s)
#print(t)
#plt.show()
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
print("*************KONIEC ULOHY CISLO 2**********\n\n")
print("****************ULOHA CISLO 3**************")
b=[0.0192,   -0.0185,    -0.0185,     0.0192]
a=[1.000,   -2.887,     2.7997,    -0.9113]
# print(b)
# print(len(b))
# print(a)
# print(len(a))
z,p,k=tf2zpk(b,a)

plt.figure(figsize=(6,5.25))

# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

#plt.grid(alpha=4.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
#DELETE plt.show()

print("*************KONIEC ULOHY CISLO 3**********\n\n")
print("****************ULOHA CISLO 4**************")
w,h=freqz(b,a,fs)
plt.figure()
plt.plot(w / 2 / np.pi * fs, abs(h))
plt.xlabel('Frekvence [Hz]')
plt.ylabel('Hodnota')
plt.title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
plt.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
print("*************KONIEC ULOHY CISLO 4**********\n\n")
print("****************ULOHA CISLO 5**************")
plt.figure()
ss = lfilter(b, a, s)
ssshifted=np.roll(ss,-15)#bolo 24

plt.xlabel('Čas [s]')
plt.ylabel('Hodnota')

plt.plot(ssshifted,'g')
plt.plot(s,'r')

print("*************KONIEC ULOHY CISLO 5**********\n\n")
print("****************ULOHA CISLO 6**************")
#s=s[:32000]

prot2=list()
i=0
while (i< len(ssshifted) ):
    i=i+8
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    if(ssshifted[i]>0):
        prot2.append(1)

    if(ssshifted[i]<0):
        prot2.append(0)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    prot2.append(None)
    i=i+8
#forprotoc=np.arange(s.size)/fs
t=np.arange(s.size)/fs
t2=t
tprot=t[:320]
plt.figure()
plt.ylabel('s[n],ss[n], ssshifted[n], symbols')
plt.xlabel('t[s]')
forprotoc=np.arange(320)
ssshiftedprot=ssshifted[:320]
prot2prot=prot2[:320]
sprot=s[:320]
ssprot=ss[:320]


plt.plot(tprot,ssprot,'b')
plt.plot(tprot,sprot,'y')
plt.scatter(tprot,prot2prot)
plt.plot(tprot,ssshiftedprot,'g')
print("*************KONIEC ULOHY CISLO 6**********\n\n")
print("****************ULOHA CISLO 7**************")
count1a=0
i=0
while(i<len(ss)):
    if(prot2[i]!=prot3[i]):
        count1a=count1a+1
    i=i+1
print(count1a)
miss=count1a/(len(prot3)/16)
miss=miss*100
print("Symboly dekodovane z ssshifted[n] maju oproti symbolom z s[n] chybovosť: ",miss,"%")

print("*************KONIEC ULOHY CISLO 7**********\n\n")
print("****************ULOHA CISLO 8**************")
ft1=abs(np.fft.fft(ss))
ft2=abs(np.fft.fft(s))
ft1=ft1[:16000]
ft2=ft2[:16000]
plt.figure()
plt.ylabel('Magnitude')
plt.xlabel('f[Hz]')
plt.plot(ft2,'r',linewidth=0.1)
plt.plot(ft1,'g',linewidth=0.1)

print("*************KONIEC ULOHY CISLO 8**********\n\n")
print("****************ULOHA CISLO 9**************")
hist, x = np.histogram(s, 50)
plt.figure()
#plt.add_subplot(111)
plt.ylabel('Value')
plt.xlabel('x')
plt.plot(x[:-1], hist/sum(hist))
plt.title("Hustota rozdelenia pravdepodobnosti")
print(sum(hist/32000))
print("*************KONIEC ULOHY CISLO 9**********\n\n")
print("****************ULOHA CISLO 10**************")
#
k = np.arange(-len(s) + 1, len(s))
Rv = np.correlate(s, s, 'full') / len(s)
plt.figure()
plt.ylabel('Value')
plt.xlabel('k')
plt.plot(k,Rv)
#plt.savefig("exercise 10")
#plt.cla()
print("*************KONIEC ULOHY CISLO 10**********\n\n")
print("****************ULOHA CISLO 11**************")
#11
print("Rv[0] :", Rv[32000])
print("Rv[1] :", Rv[32001])
print("Rv[16] :", Rv[32016])
print("*************KONIEC ULOHY CISLO 11**********\n\n")
print("****************ULOHA CISLO 12**************")
num_bins = 50
conts, edg_x, edg_y = np.histogram2d(s[:-1], s[1:]*-1, num_bins, normed=True)
sizeof_cont = edg_x[1] - edg_x[0]
srfc_cont = sizeof_cont**2

plt.figure()

plt.ylabel('x1')
plt.xlabel('x2')

plt.title("Časový odhad združenej funkcie hustoty \nrozdelenia pravdepodobnosti pre n a n+1")
plotvar = plt.imshow(conts, interpolation='nearest', extent=[edg_x[-1], edg_x[0], edg_y[-1], edg_y[0]])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

cbar = plt.colorbar(plotvar)

print("*************KONIEC ULOHY CISLO 12**********\n\n")
print("****************ULOHA CISLO 13**************")
print((sum(edg_x)+sum(edg_y))*32000)
print(sum(edg_y)*srfc_cont)
#print(conts*32000)
print("*************KONIEC ULOHY CISLO 13**********\n\n")

#plt.show()
input("Stlacte lubovolnu klavesu pre ukoncenie programu")
