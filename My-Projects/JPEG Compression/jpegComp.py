import matplotlib.pyplot as plt
import numpy as np
import imageio

'''
------------------------------------------------------------------------------------------------------------------------
|                                                                                 |
|                                                   JPEG Compression                                                        |
|                                        Henrik Skaanes Steigard (henrisst)                                            |
|                                                                                                                      |
------------------------------------------------------------------------------------------------------------------------
'''


class JPEG:


    def __init__(self, image, q):

        self.img = imageio.imread(image, as_gray=True)
        self.q = q


    def substract(self):

        # subtraherer 128 fra alle pikselintensiteter i input-bildet
        return self.img - 128


    def blocks(self, img):

        # dimensjonene til input-bildet: vertikal og horisontal dimensjon
        N, M = self.img.shape

        # vi kontrollerer at input-bildet er faktorisertbart med heltallet 8
        if N % 8 != 0 or M % 8 != 0:
            print("Dette programmet tar kun bilder med bredde og hoeyde av multipler paa 8!")

        # maaten 8 x 8 blokksystemet er implementert, er ved aa gi én unik ID til hver 8 x 8 blokk.
        # denne ID-en legges i en ordbok som nokkelverdi, mens den tilhorende verdien er den spesifiserte,
        # unike 8 x 8 blokken

        # her vil 'img_block_id' vaere en "krympet" versjon av input-bildet, hvor hver posisjon i 2D-arrayet
        # vil lagre den unike ID-en til hver unike 8 x 8 blokk som horer til input-bildet. Paa denne maaten
        # vil vi kunne iterere gjennom 2D-arrayet og hente ut hver eneste blokk som horer til input-bildet,
        # og det paa riktig posisjon i det opprinnelige input-bildet
        img_block_id = np.zeros((N//8, M//8))

        # dimensjonene av 2D-arrayet som skal lagre 8 x 8 blokkene sine posisjoner i input-bildet
        Q, R = img_block_id.shape

        # 'block_dict' vil lagre den unike ID-en som nokkel og den unike blokken som verdi. Den unike ID-en
        # angir den unike posisjonen som sentrum, eller origo, av blokken har i det originale input-bildet
        block_dict = {}

        # k er den unike ID-en tilhorende hver individuelle 8 x 8 blokk
        k = 0

        # vi oppretter en unik ID for alle blokker. Husk at Q og R er dimensjonene til 2D-blokk arrayet
        for i in range(Q):
            for j in range(R):

                # hvert unike origo til hver blokk faar sin tilhorende unike ID k - et heltall
                img_block_id[i][j] = k
                k += 1

        # vi begynner naa prosessen med aa opprette alle 8 x 8 blokker fra input-bildet
        for i in range(N//8):
            for j in range(M//8):

                # for aa hente ut alle 8 x 8 blokker, benytter vi slicing av 2D-arrayet til input-bildet
                # derfor vil vi ha en x- og y-verdi for henholdsvis start- og sluttkoordinat for hvor
                # hver blokk skal hentes ut fra det originale input-bildet sitt 2D-array
                from_x = i * 8
                to_x = (i + 1) * 8

                from_y = j * 8
                to_y = (j + 1) * 8

                # 'placement' definerer den unike ID-en som tilhorer den naavaerende blokken
                # dette gir mening fra forrige for-lokke, hvor vi satte nettopp en tilknyttet
                # ID for hvert unike origo av hver 8 x 8 blokk
                placement = img_block_id[i][j]

                # her kommer slicing - vi 'plukker' ut hver 8 x 8 blokk fra 2D-arrayet til input-bildet
                block = img[from_x:to_x, from_y:to_y]

                # blokken plasseres som verdi til den tilhorende ID-en som nokkel
                block_dict[placement] = block

        return img_block_id, block_dict


    def DCT(self, img):

        # vi starter prosessen med DCT-transform ved aa dele det spesifiserte input-bildet opp i 8 x 8 blokker
        block_id, block_dict = self.blocks(img)

        # dimensjonene til 'blokk-arrayet', altsaa en 'krympet' versjon av input-bildet hvor hver posisjon tilsvarer
        # en 8 x 8 blokk
        Q, R = block_id.shape

        # dimensjonene til det originale input-bildet
        N, M = self.img.shape

        # et tomt 2D-array som skal angi det rekonstruerte bildet med samme dimensjoner som input-bildet
        # etter at DCT-transformen er blitt gjort for hver individuelle 8 x 8 blokk
        img_rcstrd = np.zeros((N, M))

        # vi begynner med aa hente ut hver 8 x 8 blokk tilghorende det originale input-bildet
        for i in range(Q):
            for j in range(R):

                # henter ID for naaveraende 8 x 8 blokk
                # du husker kanskje fra for at hver unike ID tilhorende hver individuelle 8 x 8 blokk tilsvarer
                # denne 8 x 8 blokken sitt sentrum, eller origio, i det originale input-bildet?
                # derav:
                current_id = block_id[i][j]

                # vi kan naa hente blokken fra ordboken med spesifisert ID
                current_block = block_dict.get(current_id)

                # vi vil naa gaa gjennom hvert eneste koordinat med hver sine graatoneintensitet i den spesifiserte
                # 8 x 8 blokken
                x_cord = i * 8
                for u in range(current_block.shape[0]):

                    y_cord = j * 8
                    for v in range(current_block.shape[1]):

                        # 'out' tilsvarer den 'nye' blokken etter en DCT-transformasjon av den naavaerende 8 x 8 blokken
                        out = np.zeros((8,8))

                        # setter Cu og Cv slik det oppgis i oppgavespesifikasjonen
                        cu = 1
                        cv = 1

                        # DCT-summen settes forelopig lik 0 for gjennomgang av alle pikselintensitetene i blokken
                        sum = 0

                        if u == 0:
                            cu = 1/np.sqrt(2)

                        if v == 0:
                            cv = 1/np.sqrt(2)

                        # vi traverserer gjennom alle pikselintensiteter i blokken
                        for x in range(8):
                            for y in range(8):

                                grey_level = current_block[x][y]
                                dct = grey_level * np.cos(((2 * x) + 1) * u * np.pi / 16.0) \
                                      * np.cos(((2 * y)+ 1) * v * np.pi / 16.0)

                                sum += dct

                        # DCT-transformen for hver pikselintensitet i input-blokken
                        out[u][v] = 0.25 * cu * cv * sum

                        # DCT-transformen for hver pikselintensitet i 8 x 8 blokken settes inn i det rekonstruerte
                        # 2D-arrayet som utgjor hele DCT-transformen av input-bildet
                        img_rcstrd[x_cord][y_cord] = out[u][v]

                        y_cord += 1
                    x_cord += 1

        return img_rcstrd


    def IDCT(self, img):

        # vi deler det spesifiserte input-bildet opp i blokker paa storrelse 8 x 8
        block_id, block_dict = self.blocks(img)

        # dimensjonene til 2D-blokkarrayet, altsaa det 'krympede' 2D-arrayet som tilsvarer alle 8 x 8 blokker
        Q, R = block_id.shape

        # dimensjonene til det originale input-bildet
        N, M = self.img.shape

        # det rekonstruerte bidet etter IDCT-tramnsformasjonen
        img_rcstrd = np.zeros((N, M))

        # vi henter ut hver eneste 8 x 8 blokk for aa gjore en IDCT-transformasjon paa blokken
        for i in range(Q):
            for j in range(R):

                # henter ID for naaveraende 8 x 8 blokk
                current_id = block_id[i][j]

                # henter blokken med den tilsvarende unike ID-en
                current_block = block_dict.get(current_id)

                # vi gaar gjennom hvert eneste koordinat i den naavarende 8 x 8 blokken
                x_cord = i * 8
                for x in range(current_block.shape[0]):

                    y_cord = j * 8
                    for y in range(current_block.shape[1]):

                        # summen av IDCT-transformasjonen settes lik 0 ved start
                        sum = 0

                        # vi oppretter en ny 'blokk' for aa lagre IDCT-transformasjonen av den naavaerende blokken
                        out = np.zeros((8, 8))

                        # vi traverserer gjennom hele blokken for hver eneste pikselintensitet i denne
                        # naavarende blokken
                        for u in range(8):

                            for v in range(8):

                                # Cu og Cv settes etter oppgavespesifikasjonen
                                cu = 1
                                cv = 1

                                if u == 0:
                                    cu = 1 / np.sqrt(2)

                                if v == 0:
                                    cv = 1 / np.sqrt(2)

                                grey_level = current_block[u][v]

                                dct = cu * cv * grey_level * np.cos(((2 * x) + 1) * u * np.pi / 16.0) * \
                                      np.cos(((2 * y) + 1) * v * np.pi / 16.0)

                                sum += dct

                        # vi setter IDCT-transformen for den naavarende pikselintensiteten i riktig posisjon i
                        # den 'nye' blokken
                        out[x][y] = 0.25 * sum

                        # vi setter IDCT-transformen for den naavarende pikselen inn i det rekonsturerte bildet
                        img_rcstrd[x_cord][y_cord] = out[x][y]

                        y_cord += 1
                    x_cord += 1

        # vi legger 128 til hver eneste pikselintensitet i det rekonstruerte bildet
        img_rcstrd = img_rcstrd + 128

        # avrunder alle pikselverdier til naemeste heltallsverdi grunnet unoyaktig flyttallsartimetikk
        img_rcstrd = np.round(img_rcstrd)

        return img_rcstrd


    def quantization(self, img, divide=True):

        # vi deler input-bildet opp i 8 x 8 blokker
        block_id, block_dict = self.blocks(img)

        # den oppgitte kvantifiseringsmatrisen fra oppgavespesifikasjonen
        quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                        [12, 12, 14, 19, 26, 58, 60, 55],
                                        [14, 13, 16, 24, 40, 57, 69, 56],
                                        [14, 17, 22, 29, 51, 87, 80, 62],
                                        [18, 22, 37, 56, 68, 109, 103, 77],
                                        [24, 35, 55, 64, 81, 104, 113, 92],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103, 99]])

        # vi multipliserer kvatifiseringsmatrisen med instansverdien q
        quantization_matrix = quantization_matrix * self.q

        # dimensjonene til 2d-blokk arrayet (spesifisert fra tidligere)
        Q, R = block_id.shape

        # dimensjonene til input-bildet
        N, M = self.img.shape

        # det rekonstruerte bildet angitt ved et 2D-array
        img_rcstrd = np.zeros((N, M))

        for i in range(Q):
            for j in range(R):

                # henter ID for naaveraende 8 x 8 blokk
                current_id = block_id[i][j]

                # henter blokken med samsvarende unike ID
                current_block = block_dict.get(current_id)

                if divide == True:

                    # 'rekvantifiserer' naavaerende blokk
                    current_block = current_block / quantization_matrix
                    current_block = np.round(current_block, 0)

                elif divide == False:

                    # kvantifiserer naavarende blokk
                    current_block = current_block * quantization_matrix
                    current_block = np.round(current_block, 0)

                # vi itererer gjennom hele blokken og plasserer hver pikselintensitet paa den tilhorende posisjonen
                # i det rekonstruerte 2D-arrayet
                x_cord = i * 8
                for u in range(current_block.shape[0]):

                    y_cord = j * 8
                    for v in range(current_block.shape[1]):

                        # det rekonstruerte bildet 'bygges' piksel for piksel
                        img_rcstrd[x_cord][y_cord] = current_block[u][v]

                        y_cord += 1
                    x_cord += 1

        return img_rcstrd


    def verify_rcstrd_img(self, img_IDCT):

        N, M = self.img.shape

        for i in range(N):
            for j in range(M):

                # kontrollerer at alle pikselintensiteter i dent originale input-bildet er identistiske med det
                # IDCT-transformerte bilder
                if self.img[i][j] != img_IDCT[i][j]:

                    print("The reconstructed image is NOT equal to the original input image!")
                    exit(1)

        print("\n **The reconstructed image from the IDCT-transform and the original input image are identical.\n")


    def calculate_entropy(self, img):

        # vi antar at input-bildet har entropi tilsvarende 8, selv om dette antagligvis ikke er helt riktig
        img_org_entrp = 8

        # setter forelopig entropi for det DCT-kvantifiserte bildet lik 0
        img_DCT_qtz_entrp = 0

        N, M = self.img.shape

        # vi benytter av oss en ordbok for aa slippe unodige for-lokker naar vi itererer gjennom alle pikselverdier
        p_i = {}

        for i in range(N):
            for j in range(M):

                if img[i][j] not in p_i.keys():
                    p_i[img[i][j]] = 1

                else:
                    p_i[img[i][j]] = p_i.get(img[i][j]) + 1

        for key in p_i.keys():
            p = p_i.get(key) / (N * M)
            img_DCT_qtz_entrp += p * np.log2(p)


        img_DCT_qtz_entrp = - img_DCT_qtz_entrp

        # det "omtrentelige" lagringsbehoved for det DCT-kvantifiserte input-bildet
        improved_data_usage = (img_DCT_qtz_entrp * N * M / 8) / 1024

        # det "omtrentelige" lagringsbehoved for det originale input-bildet
        non_improved_data_usage = (img_org_entrp * N * M / 8) / 1024

        # prosentvis forbedring av dataforbruk
        improvement_prct = 100 - (np.round(improved_data_usage / non_improved_data_usage, 2) * 100)

        # det faste antall biter per symbol i den ukomprimerte datamengden
        b = 8

        # det gjennomsnittelige antall biter per symbol i den komprimerte datamengden
        c = img_DCT_qtz_entrp

        # kompresjonsraten
        CR = np.round(b / c, 4)

        print("**The DCT-quantized input image achieved an entropy of ", np.round(img_DCT_qtz_entrp, 3),
              ", which yields the approx. data usage of ", np.round(improved_data_usage, 3), "KiB")

        print("\n**The original input image achieved an entropy of ", np.round(img_org_entrp, 3),
              ", which yields the approx. data usage of ", np.round(non_improved_data_usage, 3), "KiB")

        print("\n**The JPEG compression achieved ", improvement_prct, "% less data storage needed to store the\n"
              "original input image. The current JPEG compression yields a compression rate CR of ", CR, ".")


    def view_image(self, img=None, title="No title specified"):

        if img is None:
            img = self.img

        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()


    def save(self, image, fileName):
        imageio.imwrite(fileName, image)


def main():

    # steg 1 |----------------------------------------------------------------------------------------------------------

    # bildet vi onsker aa lese inn
    image = "uio.png"

    # kvantifiseringsfaktoren
    q = 2

    # vi leser forst inn input-bildet 'uio.png' som forsteparameter og setter angitt tallparameter q som andreparameter
    img = JPEG(image, q)

    # vi kontrollerer at input-bildet leses inn korrekt ved aa vise input-bildet
    # None angir at vi vil vise input-bildet som ble angitt som instansvariabel ved opprettelse av objektet 'img'
    img.view_image(None, "Input image")

    # steg 2 |----------------------------------------------------------------------------------------------------------

    # vi subtraherer 128 fra alle pikselverdier slik at den gjennomsnittlige intensitetsveriden ligger paa omtrent 0
    img_sub = img.substract()

    # vi viser det subtraherte input-bildet
    img.view_image(img_sub, "The original input image subtracted by 128\n for each individual pixel intensity")

    # steg 3 |----------------------------------------------------------------------------------------------------------

    # vi beregner DCT-transformen av input-bildet
    img_dct = img.DCT(img_sub)

    # vi viser det DCT-transformerte bildet
    img.view_image(img_dct, "The DCT transformation of 8 x 8 blocks\n of pixels from the input image")

    # steg 4 |----------------------------------------------------------------------------------------------------------

    # vi rekonsturerer det DCT-transformerte bildet
    img_rcstrd = img.IDCT(img_dct)

    # vi programmatisk verifiserer at det rekonstruerte input-bildet fra IDCT-transformen er identisik med input-bildet
    img.verify_rcstrd_img(img_rcstrd)

    # steg 5 |----------------------------------------------------------------------------------------------------------

    # vi utforer kvantifiseringen av det DCT-transformerte input-bildet
    img_qtz = img.quantization(img_dct)

    # steg 6 |----------------------------------------------------------------------------------------------------------

    # vi finner entropien og "tilsvarende" lagringsbehov for det DCT-kvantifiserte bildet
    img.calculate_entropy(img_qtz)

    # steg 7 |----------------------------------------------------------------------------------------------------------

    # vi "dekvantifiserer" det kvantifiserte input-bildet
    img_reqtz = img.quantization(img_qtz, False)

    # vi rekonstruerer det originale input-bildet ved IDCT-transformen
    img_cpd = img.IDCT(img_reqtz)

    # vi viser resultatbildet etter den spesifiserte JPEG-komprimeringen
    img.view_image(img_cpd, "The compressed JPEG image with q = " + str(q))

    # vi kan velge aa lagre det spesifiserte input-bildet med onsket navn og filtype
    # img.save(img_cpd, "uio_" + str(q) + ".png")


if __name__ == '__main__':

    main()
