# ttadapter-od

## Prepare Dataset

Please follow dataset structure below.

1. Coco & coco-corruption
    ```
    - coco
        - train2017
        - val2017
        - val2017-snow
        - val2017-frost
        ...
    ```

2. SHIFT
    ```
    - shift
        - discrete
            - images
                - train
                    - front
                        - images
                            ...
                        - det_2d.json
                        - seq.csv
                - val
                    - front
                        ...
        - continuous1x
        - continuous10x
    ```

## Selective SHIFT Dataset NOTICE

쉘스크립트에 아래 config를 추가해 원하는 조건에 맞는 SHIFT Dataset 시퀀스를 골라 사용할 수 있습니다.

Eval-only 조건에서만 동작하며, 조건에 맞는 시퀀스가 여러 개 있더라도 하나의 시퀀스만 랜덤으로 선택됩니다.

    
    SHIFT config usage: 

        아래 선택지 중 입력. 현재 복수 선택 불가능. 따로 선택하지 않고자 할 경우 None 입력 혹은 config 선언하지 않음.
        DATASETS.SHIFT.SHIFT_TYPE = daytime_to_night, clear_to_rainy, clear_to_foggy
        DATASETS.SHIFT.WEATHER = overcast, clear, cloudy, foggy, rainy
        DATASETS.SHIFT.TIME = day, night, dawn/dusk
    


