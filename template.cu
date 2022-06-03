// libs NV
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// libs generiques
#include <math.h>

// lib spec
#include "kernelMedian.cu"

// formats des images
#define UCHAR (unsigned char)0x10
#define USHORT (unsigned short)0x1000

// longueur max ligne dans fichier pgm
#define SIZE_LINE_TEXT 256

void checkError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Erreur CUDA : %s: %s.\n", msg,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * \fn int type_image_ppm(int *prof, int *i_dim, int *j_dim, int *level, char *file_name)
 * \brief Fonction qui renvoie le type de l'image ppm et des caracteristiques
 *
 * \param[out] prof profondeur de l'image 1 pour pgm 3 pour ppm, 0 sinon
 * \param[out] i_dim renvoie la dimension verticale de l'image (si NULL, renvoie que prof)
 * \param[out] j_dim renvoie la dimension horizontale de l'image
 * \param[out] level renvoie la dynamique de l'image
 * \param[in]  file_name fichier image
 *
 * \return 1 si ok O sinon
 *
 */
int type_image_ppm(int *prof, unsigned int *i_dim, unsigned int *j_dim, int *level, char *file_name)
{
    char buffer[SIZE_LINE_TEXT];
    FILE *file;

    *prof = 0;

    file = fopen(file_name, "rb");
    if (file == NULL)
        return 0;

    // lecture de la premiere ligne
    fgets(buffer, SIZE_LINE_TEXT, file);

    /* pgm */
    if ((buffer[0] == 'P') & (buffer[1] == '5'))
        *prof = 1; // GGG
    /* ppm */
    if ((buffer[0] == 'P') & (buffer[1] == '6'))
        *prof = 3; // RVBRVBRVB

    /* type non gere */
    if (*prof == 0)
        return 0;

    /* pour une utilisation du type */
    /* ret = type_image_ppm(&prof, NULL, NULL, NULL, file_name) */
    if (i_dim == NULL)
        return 1;

    /* on saute les lignes de commentaires */
    fgets(buffer, SIZE_LINE_TEXT, file);
    while ((buffer[0] == '#') | (buffer[0] == '\n'))
        fgets(buffer, SIZE_LINE_TEXT, file);

    /* on lit les dimensions de l'image */
    sscanf(buffer, "%d %d", j_dim, i_dim);
    fgets(buffer, SIZE_LINE_TEXT, file);
    sscanf(buffer, "%d", level);

    fclose(file);
    return 1;
}

/**
 * \fn void load_pgm2int(int **image, int i_dim, int j_dim,
 *         int nb_level, char *fichier_image)
 * \brief lecture pgm 8 ou 16 bits
 *
 * \param[out] image
 * \param[in]  i_dim dimension verticale de l'image
 * \param[in]  j_dim dimension horizontale de l'image
 * \param[in]  nb_level dynamique de l'image
 * \param[in]  fichier_image fichier image
 *
 *
 */
template <class IMG_TYPE>
void load_pgm2uw(IMG_TYPE *image, int i_dim, int j_dim, char *fichier_image)
{
    int i, j;
    char buffer[SIZE_LINE_TEXT];
    FILE *file = fopen(fichier_image, "rb");

    fgets(buffer, SIZE_LINE_TEXT, file); /* P5 */
    /* on saute les lignes de commentaires */
    fgets(buffer, SIZE_LINE_TEXT, file);
    while ((buffer[0] == '#') | (buffer[0] == '\n'))
        fgets(buffer, SIZE_LINE_TEXT, file);
    /* derniere ligne lue : dimensions */
    fgets(buffer, SIZE_LINE_TEXT, file); /* dynamique */

    /* data */
    // fichier en char ou en short, on convertit selon
    IMG_TYPE *ligne;
    ligne = (IMG_TYPE *)malloc(sizeof(IMG_TYPE) * j_dim);
    for (i = 0; i < i_dim; i++)
    {
        fread(ligne, sizeof(IMG_TYPE), j_dim, file);
        for (j = 0; j < j_dim; j++)
            image[i * j_dim + j] = (IMG_TYPE)(ligne[j]);
    }
    free(ligne);
    fclose(file);
}

template <class IMG_TYPE>
void save_2pgm(char *fichier_image, IMG_TYPE *image, int j_dim, int i_dim)
{
    int i, j;
    FILE *file = fopen(fichier_image, "wb");

    // entete pgm
    // format
    fprintf(file, "P5\n");
    fprintf(file, "# AND - DISC\n");
    fprintf(file, "# FEMTO-ST Institute - Belfort - France\n");
    // taille
    fprintf(file, "%d %d\n", j_dim, i_dim);
    // dynamique
    unsigned short dyn = (1 << sizeof(IMG_TYPE) * 8) - 1;
    printf("save2pgm dyn=%d\n", dyn);
    IMG_TYPE *ligne;
    fprintf(file, "%d\n", dyn);
    ligne = (IMG_TYPE *)malloc(sizeof(IMG_TYPE) * j_dim);
    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
            ligne[j] = (IMG_TYPE)(image[i * j_dim + j]);
        fwrite(ligne, sizeof(IMG_TYPE), j_dim, file);
    }
    free(ligne);
    fclose(file);
}

template <class IMG_TYPE>
void run_test(int argc, char **argv, int r, int ppt, IMG_TYPE flag)
{

    // assuming this is the one
    int gpuId = 0;
    if (argc > 2)
        gpuId = atoi(argv[1]);
    cudaSetDevice(gpuId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuId);
    // for time measurements
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // input file name
    char *image_name = argv[argc - 1];
    // CPU
    IMG_TYPE *h_datac = NULL;       // primary input image
    IMG_TYPE *h_datacPadded = NULL; // padded memory area for input image
    unsigned int H, L, size;        // image dimensions and size
    IMG_TYPE *h_out_gpu_c;          // output image ready for hard disk write
    dim3 dimBlock, dimGrid;         // grid dimensions
    int bsx = 32, bsy = 8;          // default values of thread block dimensions
    int ct = 0;                     // counter for mean execution time

    int profImage;
    int depth;
    int bitDepth;

    // GPU data output
    IMG_TYPE *d_outc;
    IMG_TYPE *d_inc;

    // dummy alloc to avoid possible high latency when first using the GPU
    short *d_bidon;
    cudaMalloc((void **)&d_bidon, sizeof(short));

    if (type_image_ppm(&profImage, &H, &L, &depth, image_name))
    {
        bitDepth = log2(1.0 * (depth + 1));
        // loads image from hard disk
        sdkStartTimer(&timer);
        h_datac = (IMG_TYPE *)malloc(H * L * sizeof(IMG_TYPE));
        // sdkLoadPGM(image_name, &h_datac, &L, &H);
        load_pgm2uw(h_datac, H, L, image_name);
        sdkStopTimer(&timer);

        // memory size of the image (CPU side)
        size = H * L * sizeof(IMG_TYPE);

        // loading summary
        printf("\n***** CONVOMED SUMMARY *****\n");
        printf("GPU : %s\n", prop.name);
        printf("Image %d bits %s  (%d x %d) pixels = %d Bytes loaded in %f ms,\n", bitDepth, image_name, L, H, size, sdkGetTimerValue(&timer));
    }

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // GPU memory allocations
    int Hpitch;
    checkCudaErrors(cudaMalloc((void **)&d_outc, H * L * sizeof(IMG_TYPE)));
    checkError("Alloc dout_c");
    checkCudaErrors(cudaMallocPitch((void **)&d_inc, (size_t *)&Hpitch, (size_t)((L + 2 * r) * sizeof(IMG_TYPE)), (size_t)((H + 2 * r) * sizeof(IMG_TYPE))));
    checkError("Alloc d_inc");
    sdkStopTimer(&timer);
    printf("GPU memory allocations done in %f ms\n", sdkGetTimerValue(&timer));

    // PAGED LOCKED mem
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    checkCudaErrors(cudaHostAlloc((void **)&h_out_gpu_c, H * L * sizeof(IMG_TYPE), cudaHostAllocDefault));
    h_datacPadded = (IMG_TYPE *)malloc((H + 2 * r) * Hpitch * sizeof(IMG_TYPE));
    if (h_datacPadded != NULL)
        printf("ALLOC padded mem CPU OK\n");
    sdkStopTimer(&timer);
    printf("CPU memory allocations done in %f ms\n", sdkGetTimerValue(&timer));

    int i, j;
    int h_dim = Hpitch / sizeof(IMG_TYPE);
    for (i = 0; i < H; i++)
        for (j = 0; j < L; j++)
            h_datacPadded[(i + r) * h_dim + j + r] = 1.0 * h_datac[i * L + j];

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    checkCudaErrors(cudaMemcpy(d_inc, h_datacPadded, (H + 2 * r) * Hpitch, cudaMemcpyHostToDevice));
    checkError("Copie h_datac en GMEM --> d_inc");

    sdkStopTimer(&timer);
    printf("Input image copied into global memory in %f ms\n", sdkGetTimerValue(&timer));

    sdkResetTimer(&timer);

    /*****************************
     * Kernels calls
     *****************************/
    checkError("Config cache");
    dimBlock = dim3(bsx, bsy, 1);
    dimGrid = dim3((L / dimBlock.x) / 1, (H / dimBlock.y) / ppt, 1);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    // 3 different kernels matrixMedianFilterNaive(), matrixMedianFilterOptimized(), matrixMedianFilterAdvancedOptimized()
    for (ct = 0; ct < 100; ct++)
        matrixMedianFilterOptimized<<<dimGrid, dimBlock, 0>>>(d_inc, d_outc, L, h_dim);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    printf("Mean runtime (on %d executions): %f ms - soit %.0f Mpixels/s \n", (ct), sdkGetTimerValue(&timer) / (ct), L * H / (1000.0 * sdkGetTimerValue(&timer) / (ct)));

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    checkCudaErrors(cudaMemcpy((void *)h_out_gpu_c, d_outc, H * L * sizeof(IMG_TYPE), cudaMemcpyDeviceToHost));
    checkError("Copie D_out depuis GMEM vers mem CPU");

    sdkStopTimer(&timer);
    printf("Ouput image (image_out.pgm) copied from GPU to CPU in %f ms\n", sdkGetTimerValue(&timer));
    printf("***** END OF CONVOMED EXECUTION *****\n\n");

    // sdkSavePGM("image_out.pgm", h_out_gpu_c, L, H) ;
    save_2pgm((char *)("image_out.pgm"), h_out_gpu_c, L, H);

    checkError("Writing img on disk");

    cudaFree(d_inc);
    cudaFree(d_outc);
    cudaFreeHost(h_out_gpu_c);
}

int main(int argc, char **argv)
{
    std::cout << "Main Started" << std::endl;
    // mask radius
    int r = 2;
    // pixels per thread
    int ppt = 2;
    unsigned int H, L;
    int profImage, depth, bitDepth;
    char *image_name = argv[argc - 1];

    if (type_image_ppm(&profImage, &H, &L, &depth, image_name))
    {
        bitDepth = log2(1.0 * (depth + 1));
        switch (bitDepth)
        {
        case 8:
            run_test(argc, argv, r, ppt, UCHAR);
            break;
        case 16:
            run_test(argc, argv, r, ppt, USHORT);
            break;
        }
    }
    return EXIT_SUCCESS;
}
