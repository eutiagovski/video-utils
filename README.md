# IA Video Utils

Este projeto é um utilitário que visa economizar tempo em pesquisas relacionadas à visão computacional. 

Compilamos algumas das principais ferramentas relacionadas da área em uma ferramenta simples e fácil de utilizar.

Você pode utilizar um dos modelos já treinados ou integrar facilmente um novo modelo.
<br>

# Para que serve este projeto?

Você pode executar este projeto para executar as seguintes ações:

- Transmissão diretamente da webcam, ip-cameras e outros
- Salvar uma foto diretamente da transmissão 
- Timelapse (pode configurar o total de fotos e tempo de intervalo)

- Detecção de objetos (Atualmente usando a SSD MobileNet V2)
- Detecção de rostos (Atualmente usando o Caascade)
- Detecção de rosto com emoji (Atualmente usando a Deeperface)
 
- Detecção de movimento (Built in)
- Detecção de Objetos Avançada (Mask-RCNN)

<br>


# Como utilizar este projeto?

1) Faça um clone do repositório para a sua máquina
2) Instale as depenpendências 

3) Importe a bilbioteca 
4) Instancie a ferramenta
5) Começe a transmissão

<br>

# Funções Presentes na Ferramenta:

- d - Detecta objetos no video (por padrão detecta rosto, mas pode atualizado alterando-se o modelo)
- e - Detecta emoções com emoji

- m - Detecta movimentos no video
- n - Altera o modelo de detecção de objeto

- s - Salva a imagem na pasta raiz do projeto

## Detecção de Rostos com Emoji 

<img src="./docs/images/Screenshot from 2022-10-16 12-08-56.png" alt="Detect Objvects MaskRCNN" />

<br>

## Detecção de Movimento 

<img src="./docs/images/Screenshot from 2022-10-16 12-52-15.png" alt="Detect Objvects MaskRCNN" />

<br>

## Detecção de Objetos MaskRcnn

<img src="./docs/images/Screenshot from 2022-10-16 12-51-47.png" alt="Detect Objvects MaskRCNN" />

<br>


# Próximos Passos

- Usuário criar boundbox de referência para padronização de imagens
- Desenvolvimento da função callback com o frame e as informações 

# Tenologias

- Python
- OpenCV
- Tensorflow

# Licença

<a href='./LICENSE'>MIT 2022 @ Tiago de Oliveira Machado
(tiagomachadodev@gmail.com)</a>

# Referencias:

https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/