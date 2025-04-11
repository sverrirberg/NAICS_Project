# NAICS Flokkunarkerfi

Þetta forrit notar OpenAI API til að þýða innkaupalýsingar og reikningsflokkanir yfir á ensku, og síðan flokka þær eftir NAICS kerfinu.

## Uppsetning

1. Klónið þetta verkefni
2. Búið til og virkjið sýndarumhverfi:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Á Windows: venv\Scripts\activate
   ```
3. Setjið upp nauðsynlega pakka:
   ```bash
   pip install -r requirements.txt
   ```
4. Búið til `.env` skrá og bætið við OpenAI API lykli:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Notkun

1. Keyrið forritið:
   ```bash
   streamlit run app.py
   ```
2. Hlaðið inn CSV skrá með eftirfarandi dálkum:
   - `procurement_description`: Lýsing á innkaupum
   - `account`: Reikningsflokkun

## Niðurstöður

Forritið mun:
1. Þýða bæði dálka yfir á ensku
2. Finna viðeigandi NAICS kóða fyrir hverja línu
3. Sýna niðurstöður í töflu
4. Gefa möguleika á að hlaða niður niðurstöðum sem CSV skrá 