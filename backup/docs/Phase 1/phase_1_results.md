# Vaihe 1: Liiketoiminnan ymmärtäminen

## Liiketoiminnan tavoitteiden määrittely
Projektin kuvitteellinen asiakas on **Lontoon kaupungin energiayhtiö**, joka haluaa hyödyntää älymittaridataa sähkönkulutuksen analysointiin ja optimointiin. Tämä liittyy Euroopan unionin aloitteeseen edistää älymittarien käyttöönottoa kaikissa kotitalouksissa. EU:n ja Ison-Britannian hallituksen tavoitteena on:
- Päivittää ikääntyvä energiajärjestelmä.
- Vähentää energiankulutusta ja ympäristökuormitusta.
- Parantaa energiankäytön seurantaa ja suunnittelua.

Datasetti perustuu UK Power Networksin johtamaan **Low Carbon London** -projektiin (2011–2014), ja se sisältää älymittareiden keräämiä sähkönkulutustietoja 5,567 Lontoon kotitaloudesta sekä säädataa Darksky API:sta.

### Liiketoiminnalliset tavoitteet:
1. **Kulutuksen ymmärtäminen**: Selvitetään kotitalouksien päivittäisiä ja puolentunnin kulutusmalleja.
2. **Kulutuksen ennustaminen**: Kehitetään ennustemalli sähkönkulutuksen huippu- ja matalajaksoille.
3. **Energiatehokkuus**: Suositellaan keinoja energian säästämiseksi kotitalouksille.
4. **Resurssien optimointi**: Autetaan energiayhtiötä optimoimaan energianjakelua.

## Nykytilanteen analyysi
Lontoon energiayhtiöllä on käytössään seuraavat datalähteet:
- **Kotitaloustiedot**: Kotitalouksien ACORN-luokittelu (sosioekonomiset ja demografiset tiedot).
- **Puolentunnin mittausdata**: Sähkönkulutuksen puolentunnin mittaukset kotitalouksista.
- **Säädata**: Päivittäiset ja tuntikohtaiset sääolosuhteet.

Tämänhetkiset haasteet:
- Datan suuri määrä ja sen monimuotoisuus.
- Yhdistämisen ja analysoinnin vaativuus.
- Ennustemallien rakentamisen osaamisen puute.

## Ensisijaiset tavoitteet datan analysoinnille ja mallinnukselle
1. Tunnistetaan kulutusmalleja (esim. segmentointi päivittäisen käytön perusteella).
2. Ennustetaan sähkönkulutusta kotitalouksien ja Lontoon tasolla.
3. Yhdistetään kulutustiedot ACORN-luokitukseen ja sääolosuhteisiin syvemmän analyysin mahdollistamiseksi.
4. Luodaan simulaatioita, kuten "mitä jos" -analyyseja sähköauton latausjärjestelmien vaikutuksista.

## Projektin tavoitteiden määrittely
Projektin päätavoitteena on tuottaa seuraavat tulokset:
- **Visualisoinnit**: Sähkönkulutuksen trendit, päivittäiset vaihtelut ja poikkeavuudet.
- **Ennustemalli**: Sähkönkulutuksen tarkka ennustaminen (esim. lyhyen ja pitkän aikavälin).
- **Dataan perustuvat suositukset**: Kotitalouksille energiatehokkuuden parantamiseksi ja energiayhtiölle resurssien optimoimiseksi.
- **Skenaariotyökalu**: Simuloida, miten ulkoiset tekijät, kuten sää tai sähköautot, vaikuttavat kulutukseen.

## Tukikysymykset

### Liiketoiminnalliset tavoitteet
- Kuinka energiayhtiö voi parantaa energianjakelua?
- Miten sääolosuhteet vaikuttavat kulutushuippuihin?
- Miten kotitaloudet voivat säästää energiaa ja vähentää kulujaan?

### Projektin tulokset
- Tarkka ennustemalli.
- Kulutusmalleihin perustuvat visualisoinnit.
- Simulointityökalut ja raportti asiakkaille ja päätöksentekijöille.

### Tulosten mittaaminen
- Ennustemallien tarkkuus (MAPE, RMSE).
- Sähköhävikin väheneminen energiayhtiölle.
- Kotitalouksien saama palaute suosituksista.

### Asiakkaat ja hyödynsaajat
- **Kuvitteellinen asiakas**: Lontoon energiayhtiö.
- **Hyödynsaajat**: Kotitaloudet, energiayhtiön työntekijät, kaupungin päätöksentekijät ja ympäristö.

### Teknologiat
- **Datan käsittely**: Pandas, NumPy.
- **Visualisointi**: Matplotlib, Seaborn.
- **Ennustemallit**: Scikit-learn, TensorFlow/Keras.
- **Säädatan integrointi**: API-rajapinnat (esim. Darksky).
- **Skenaarioanalyysi**: Simulaatiotyökalut.

### Tarvittavat henkilöstötaidot
- Data-analytiikka ja koneoppiminen.
- Datan esikäsittely ja integrointi.
- Visualisoinnin ja raportoinnin osaaminen.
- Projektinhallinta ja asiakasviestintä.

### Työnjako
- **Data-insinööri**: Datan esikäsittely ja yhdistely.
- **Data-analyytikko**: Visualisointi ja analysointi.
- **Data Scientist**: Ennustemallien kehittäminen.
- **Projektipäällikkö**: Työn organisointi ja asiakasraportointi.

---
