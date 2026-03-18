## Diploma Thesis: Block Replacement Topics Overiew

The purpose of this document is to provide a logical mind map of different related topics in our diploma thesis. The document also serves as a distilled summary context for topic digestion by user as well as basic context window for AI models.

___
### Thesis Topic

TBA
___

### Thesis Research Framing
Objektom kompresie je MLP blok ako izolovaná funkcia, to znamená že replacement stratégia prebieha na úrovni bloku nie na úrovni celého modelu. LLM Model obsahujúci $n$-blokov predstavuje priestor $n$-blokových náhrad. Cieľom je analyzovať správanie bloku samotného aj správanie replacementu na singulárnej izolovanej úrovni a tak isto ako systém integrovaných náhrad. Cieľom nie je nutná náhrada všetkých blokov ale náhrada tých $k$ blokov ktorých kompresia predstavuje najväčší benefit z hľadiska trade-off analýzy podľa scenára aplikácie nového modelu.

### Block-level Replacement Methodologies

Táto oblasť predstavuje deep dive analýzu replacement stratégie izolovaného bloku, patrí sem:

- ako získať I/O páry bloku
- akú loss function na úrovni bloku
- typy model families pre replacement, e.g.:
    - Simple Linear Layer
    - Shallow MLP 
    
- deep dive do MLP a deep architektúr:
    - prečo práve structured prunning?:
        - hint: fyzický výpočet GPU neignoruje nulové váhy -> treba redukovať veľkost and/or počet váhových matíc

- čo predstavuje "dobrý" replacement:
    - analyzovať rôzne vyššie definované architektúry
    - analyzovať vnútorné správanaie modelov -> skúsiť zistiť nejaké spoločné znaky (napr. na základe správania gradientov, ...)

- 

- single block replacement:
    - infra
    - methodology
    - single-workflow
    - analysis
    - deep dive into makes a good block replacement (e.g. gradient monitoring)

### Data, Callibration & Recovery

- Knowledge distillation
- calibration I/O pairs


### Selection / Scoring / Block replacability

- stratégie skóringu:
    1. Random k-blokov
    2. "Best" k-blokov v prvej iterácií
    3. Výber najlepšieho bloku v $i$-tej iterácii (max $k$-iterácií) -> iterative scoring (najdolezitejsie bloky sa mozu menit v priebehu replacementu):
        - idea tu je taká, že propagácia chyby po náhrade bloku môže fundamentálne zmeniť správanie modelu, ak napríklad nahradíme všetkých $k$-blokov naraz

### Multi-block Workflow (Model-level)

### Model Quality, Evaluation and Trade-off Analysis

### Infrastructure, Code & Reproducibility