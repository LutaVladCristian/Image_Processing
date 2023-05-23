﻿<a name="br1"></a>Nume: Luta Vlad Crisꢀan

Tema 1

**Keras-cv:**

` `Keras-cv este un model generaꢀv de imagini bazat pe ce se inꢀtulează în limba engleză “Stable Diﬀusion”.

` `Aceasta foloseste ca input un prompt (descriere sub forma de text a ce dorim să transpunem în imagini) și

` `Limitările modelului au inceput sa reiasă în momentul în care am încercat să generăm lucruri de culori care

` `O posibilă explicație ar ﬁ faptul că portocaliul se apropie foarte mult de maroniu ca nuanță, într-un cât am

**Yolo V5:**

` `Yolo este o rețea de detecție preantrenată, uꢀlă pentru detecția obiectelor din imagini, dar și din input

` `Inferența poate ﬁ rulată încă din linie de comandă împreună cu o serie de parametrii, dintre care putem

` `Pentru acest proiect, am realizat detecția cu valori ale threshold-ului nu mai mari de cea default, adica 0,25.

` `O soluție pentru îmbunătațirea rezultatelor pe imaginile generate poate ﬁ antrenarea rețelei cu un set



<a name="br2"></a>Nume: Luta Vlad Crisꢀan

**Spațiul de culoare HSV:**

` `Spațiul de culoare HLS (Hue, Lightness, Saturaꢀon) este un sistem de reprezentare a culorilor bazat pe trei

` `Pentru a converꢀ un sistem de culoare RGB (Red, Green, Blue) în sistemul de culoare HLS (Hue, Lightness,

` `Normalizarea valorilor de roșu, verde și albastru: înainte de a efectua conversia, valorile roșu (R), verde (G)

` `Calculul valorii de luminanță (Lightness): se idenꢀﬁcă valoarea minimă și valoarea maximă dintre R, G și B.

` `Calculul valorii de saturație (Saturaꢀon): dacă L este 0 sau 1, atunci saturația este 0. Alꢁel, se calculează

` `Calculul valorii de nuanță (Hue): dacă diﬀ este 0, atunci nuanța (H) este 0. În caz contrar, se calculează nuanța

` `În ﬁnal, se normalizează nuanța asꢁel încât să ﬁe între 0 și 360 de grade: H = H \* 60 sau H = H + 360 (H este

**Aplicarea mășꢀi și transformarea într-o imagine fără fundal:**

` `Pentru crearea mășꢀi se aleg doi vectori: high si low. Ambii vectori reprezintă limita superioară, respecꢀv

` `Pe urma, aplicăm masca imaginii printr-o operaꢀe de AND logic pe biții dintre imaginea iniꢀala si masca care

` `În ﬁnal, pentru a transforma background-ul din negru în alb, iterăm prin ﬁecare pixel al imagii anterioare și