
// function on device to compare two values and swap if necessary

__device__ void comapareSwapMax(int *a, int *b)
{
    if (*a > *b)
    {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

__device__ void comapareSwapMin(int *a, int *b)
{
    if (*a < *b)
    {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

// create function on device to compare several values and swap if necessary
__device__ void comapareFirst(int *a, int *b, int *c, int *d, int *e, int *f)
{
    comapareSwapMax(a, b);
    comapareSwapMax(b, c);
    comapareSwapMax(c, d);
    comapareSwapMax(d, e);
    comapareSwapMax(e, f);
    comapareSwapMin(e, d);
    comapareSwapMin(d, c);
    comapareSwapMin(c, b);
    comapareSwapMin(b, a);
}

__device__ void comapareSecond(int *a, int *b, int *c, int *d, int *e)
{
    comapareSwapMax(a, b);
    comapareSwapMax(b, c);
    comapareSwapMax(c, d);
    comapareSwapMax(d, e);
    comapareSwapMin(d, c);
    comapareSwapMin(c, b);
    comapareSwapMin(b, a);
}

__device__ void comapareThird(int *a, int *b, int *c, int *d)
{
    comapareSwapMax(a, b);
    comapareSwapMax(b, c);
    comapareSwapMax(c, d);
    comapareSwapMin(c, b);
    comapareSwapMin(b, a);
}

__device__ void comapareForth(int *a, int *b, int *c)
{
    comapareSwapMax(a, b);
    comapareSwapMax(b, c);
    comapareSwapMin(b, a);
}

template<class IMG_TYPE>
__global__ void matrixMedianFilterNaive( const IMG_TYPE * __restrict__ inputImage, IMG_TYPE *outputImage, int imageWidth, int imageHeight)
{
    // in g_image take matrix with size 3x3 and calculate median value for the center pixel
    // and put it in g_image_m
    // Set row and colum for thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    // Take fiter window
    unsigned int filterVector[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // Deal with boundry conditions
    if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
        outputImage[row * imageWidth + col] = inputImage[row * imageWidth + col];
    else
    {
        // fill filterVector
        for (int x = 0; x < 3; x++)
        {
            for (int y = 0; y < 3; y++)
            {
                // setup the filterign window.
                filterVector[x * 3 + y] = inputImage[(row + x - 1) * imageWidth + (col + y - 1)];
            }
        }
        // sort filterVector
        for (int i = 0; i < 9; i++)
        {
            for (int j = i + 1; j < 9; j++)
            {
                if (filterVector[i] > filterVector[j])
                {
                    int temp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = temp;
                }
            }
        }

        // Set the output variables.
        outputImage[row * imageWidth + col] = filterVector[4];
    }
}

template<class IMG_TYPE>
__global__ void
matrixMedianFilterOptimized( const IMG_TYPE * __restrict__ inputImage, IMG_TYPE *outputImage, int imageWidth, int imageHeight)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row == 0) || (col == 0) || (row == imageHeight - 1) || (col == imageWidth - 1))
        outputImage[row * imageWidth + col] = inputImage[row * imageWidth + col];
    else
    {
        // get items seperatly and find the biggist
        int i0 = inputImage[(row - 1) * imageWidth + (col - 1)];
        int i1 = inputImage[(row - 1) * imageWidth + (col)];
        int i2 = inputImage[(row - 1) * imageWidth + (col + 1)];
        int i3 = inputImage[(row)*imageWidth + (col - 1)];
        int i4 = inputImage[(row)*imageWidth + (col)];
        int i5 = inputImage[(row)*imageWidth + (col + 1)];
        int i6 = inputImage[(row + 1) * imageWidth + (col - 1)];
        int i7 = inputImage[(row + 1) * imageWidth + (col)];
        int i8 = inputImage[(row + 1) * imageWidth + (col + 1)];

       // compare with each other then find and swap the biggest
        comapareFirst(&i0, &i1, &i2, &i3, &i4, &i5);
        comapareSecond(&i1, &i2, &i3, &i4, &i6);
        comapareThird(&i2, &i3, &i4, &i7);
        comapareForth(&i3, &i4, &i8);

        // set the output
        outputImage[row * imageWidth + col] = i4;
    }
}


#define SIZE 16
// matrix median filter advance optimized
template<class IMG_TYPE>
__global__ void
matrixMedianFilterAdvancedOptimized( const IMG_TYPE * __restrict__ inputImage, IMG_TYPE *outputImage, int imageWidth, int imageHeight)
{
    const int x = threadIdx.x; 
    const int y = threadIdx.y; 

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned short smem[SIZE + 2][SIZE + 2];

    //  Fill the shared memory border with zeros
    if (x == 0)
        //  left border
        smem[x][y + 1] = 0; 
    else if (x == SIZE - 1)
        //  right border
        smem[x + 2][y + 1] = 0; 
    if (y == 0)
    {
        //  upper border
        smem[x + 1][y] = 0; 
        if (x == 0)
            //  top-left corner
            smem[x][y] = 0; 
        else if (x == SIZE - 1)
            //  top-right corner
            smem[x + 2][y] = 0; 
    }
    else if (y == SIZE - 1)
    {
        //  bottom border
        smem[x + 1][y + 2] = 0; 
        if (x == 0)
            //  bottom-left corder
            smem[x][y + 2] = 0; 
        else if (x == SIZE - 1)
            //  bottom-right corner
            smem[x + 2][y + 2] = 0; 
    }

    //  Fill shared memory
    //  center
    smem[x + 1][y + 1] = inputImage[row * imageWidth + col]; 
    if ((x == 0) && ((col > 0)))
        //  left border
        smem[x][y + 1] = inputImage[row * imageWidth + col - 1]; 
    else if ((x == SIZE - 1) && (col < imageWidth - 1))
        //  right border
        smem[x + 2][y + 1] = inputImage[row * imageWidth + col + 1]; 
    if ((y == 0) && (row > 0))
    {
        //  upper border
        smem[x + 1][y] = inputImage[(row - 1) * imageWidth + col]; 
        if ((x == 0) && ((col > 0)))
            //  top-left corner
            smem[x][y] = inputImage[(row - 1) * imageWidth + col - 1]; 
        else if ((x == SIZE - 1) && (col < imageWidth - 1))
            //  top-right corner
            smem[x + 2][y] = inputImage[(row - 1) * imageWidth + col + 1]; 
    }
    else if ((y == SIZE - 1) && (row < imageHeight - 1))
    {
        //  bottom border
        smem[x + 1][y + 2] = inputImage[(row + 1) * imageWidth + col]; 
        if ((x == 0) && ((col > 0)))
            //  bottom-left corder
            smem[x][y + 2] = inputImage[(row - 1) * imageWidth + col - 1]; 
        else if ((x == SIZE - 1) && (col < imageWidth - 1))
            //  bottom-right corner
            smem[x + 2][y + 2] = inputImage[(row + 1) * imageWidth + col + 1]; 
    }
    __syncthreads();

    //  Pull the 3x3 window in a local array
    unsigned short v[9] = {smem[x][y], smem[x + 1][y], smem[x + 2][y],
                           smem[x][y + 1], smem[x + 1][y + 1], smem[x + 2][y + 1],
                           smem[x][y + 2], smem[x + 1][y + 2], smem[x + 2][y + 2]};

    //  Bubble-sort
    for (int i = 0; i < 5; i++)
    {
        for (int j = i + 1; j < 9; j++)
        {
            if (v[i] > v[j])
            { // swap
                unsigned short tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }

    //  Pick the middle one
    outputImage[row * imageWidth + col] = v[4];
}


