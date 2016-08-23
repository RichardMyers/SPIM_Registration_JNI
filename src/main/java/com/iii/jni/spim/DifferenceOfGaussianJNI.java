package com.iii.jni.spim;

import mpicbg.imglib.algorithm.scalespace.DifferenceOfGaussianPeak;
import mpicbg.imglib.algorithm.scalespace.DifferenceOfGaussianReal1;
import mpicbg.imglib.algorithm.scalespace.SubpixelLocalization;
import mpicbg.imglib.container.array.ArrayContainerFactory;
import mpicbg.imglib.cursor.LocalizableByDimCursor;
import mpicbg.imglib.cursor.LocalizableCursor;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.image.ImageFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyValueFactory;
import mpicbg.imglib.type.numeric.real.FloatType;
import mpicbg.imglib.wrapper.ImgLib2;
import mpicbg.spim.io.IOFunctions;
import mpicbg.spim.registration.detection.DetectionSegmentation;
import net.imglib2.*;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.multithreading.SimpleMultiThreading;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import spim.fiji.spimdata.interestpoints.InterestPoint;
import spim.process.interestpointdetection.Downsample;
import spim.process.interestpointdetection.ProcessDOG;

import java.util.*;

import ij.ImageJ;

/**
 * Created by Richard on 8/16/2016.
 *
 * Test for trying to reproduce via direct JNI calls the results of interactive interest point discovery via the
 * Difference-of-Gaussian routines in the FIJI plugin SPIM_Registration.
 *
 * Two methods were tried, DifferenceOfGaussian.compute and InteractiveDoG.updatePreview
 *
 */
public class DifferenceOfGaussianJNI {

    // call this method from JNI, returns array with {x,y} sub-pixel points of interest
    public static float[] compute(final float[] inImage, final int width, final int height, final int depth,
                                  final float calXY, final float calZ,
                                  final int downsampleXY, final int downsampleZ,
                                  final float sigma1, final float threshold,
                                  final boolean useInteractiveDoGMethod)
    {
        float[] interestPointsArray = null;
        try {
            if (useInteractiveDoGMethod) {
                // use example from InteractiveDoG.java
                interestPointsArray = InteractiveDoG(inImage, width, height, sigma1, threshold);
            } else {
                // use example from DifferenceOfGaussian.java
                interestPointsArray = DifferenceOfGaussian(inImage, width, height, depth,
                        calXY, calZ, downsampleXY, downsampleZ, sigma1, threshold);
            }
        } catch (Exception e) {
            IOFunctions.println("DifferenceOfGaussianJNI.compute:: failed. " + e);
            e.printStackTrace();
        }

        return interestPointsArray;
    }

    public static float[] InteractiveDoG(final float[] inImage, final int width, final int height,
                                         final float sigma1, final float threshold)
    {
        IOFunctions.println( "Using DifferenceOfGaussian method.");

        Image<FloatType> img = new ImageFactory<FloatType>(new FloatType(), new ArrayContainerFactory()).
                createImage(new int[]{width, height});

        final int[] location = new int[2];
        final LocalizableCursor<FloatType> cursor = img.createLocalizableCursor();
        final long[] pos = new long[2];
        while (cursor.hasNext()) {
            cursor.fwd();
            cursor.getPosition(location);
            final long index = location[1] * width + location[0];
            cursor.getType().set(inImage[(int) index]);
        }

        // interactive DoG
        final float k, K_MIN1_INV;
        final float[] sigma, sigmaDiff;
        int sensitivity = 4;
        float imageSigma = 0.5f;
        float thresholdMin = 0.0001f;

        k = (float) DetectionSegmentation.computeK(sensitivity);
        K_MIN1_INV = DetectionSegmentation.computeKWeight(k);
        sigma = DetectionSegmentation.computeSigma(k, sigma1);
        sigmaDiff = DetectionSegmentation.computeSigmaDiff(sigma, imageSigma);

        // the upper boundary
        float sigma2 = sigma[1];

        final DifferenceOfGaussianReal1<FloatType> dog =
                new DifferenceOfGaussianReal1<FloatType>(img, new OutOfBoundsStrategyValueFactory<FloatType>(), sigmaDiff[0], sigmaDiff[1], threshold / 4, K_MIN1_INV);
        dog.setKeepDoGImage(true);
        dog.process();

        final SubpixelLocalization<FloatType> subpixel = new SubpixelLocalization<FloatType>(dog.getDoGImage(), dog.getPeaks());
        subpixel.process();

        ArrayList<DifferenceOfGaussianPeak<FloatType>> peaks;
        peaks = dog.getPeaks();

        float[] interestPointsArray = new float[peaks.size() * 2];
        for (int i = 0; i < peaks.size(); i++) {
            interestPointsArray[i * 2] = peaks.get(i).get(0);
            interestPointsArray[i * 2 + 1] = peaks.get(i).get(1);
        }

        return interestPointsArray;
    }

    public static float[] DifferenceOfGaussian(final float[] inImage, final int width, final int height, final int depth,
                                               final float calXY, final float calZ,
                                               final int downsampleXY, final int downsampleZ,
                                               final float sigma1, final float threshold)
    {
        IOFunctions.println( "Using DifferenceOfGaussian method.");

        Img<net.imglib2.type.numeric.real.FloatType> data = ArrayImgs.floats( inImage, new long[]{ width, height, depth } );

        // down sample 'data' to create 'input'
        final AffineTransform3D affineTransform = new AffineTransform3D();
        final RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> input =
                downsample(data, calXY, calZ, downsampleXY, downsampleZ, affineTransform);

        // pre smooth data
        double additionalSigmaX = 0.0;
        double additionalSigmaY = 0.0;
        double additionalSigmaZ = 0.0;
        preSmooth(input, additionalSigmaX, additionalSigmaY, additionalSigmaZ);

        // wrap 'input' to create imglib1 'img'
        final Image<FloatType> img = ImgLib2.wrapFloatToImgLib1((Img<net.imglib2.type.numeric.real.FloatType>) data);

        ArrayList< InterestPoint >interestPoints = ProcessDOG.compute(
                null, // cuda
                null, // deviceList
                false, // accurateCUDA
                0, // percentGPUMem
                img,
                (Img<net.imglib2.type.numeric.real.FloatType>) data,
                sigma1,
                threshold,
                1, // localization = quadratic
                0.5, // imageSigmaX
                0.5, // imageSigmaY
                0.5, // imageSigmaZ
                false, // findMin
                true, // findMax
                0.0, // minIntensity
                65535.0, // maxIntensity
                false // keepIntensity
        );

        float[] interestPointsArray = new float[interestPoints.size() * 3];
        for (int i = 0; i < interestPoints.size(); i++) {
            interestPointsArray[i * 3] = interestPoints.get(i).getFloatPosition(0);
            interestPointsArray[i * 3 + 1] = interestPoints.get(i).getFloatPosition(1);
            interestPointsArray[i * 3 + 2] = interestPoints.get(i).getFloatPosition(2);
        }

        return interestPointsArray;
    }

	final public static void addGaussian(
			final Img< net.imglib2.type.numeric.real.FloatType > image,
			final double[] location,
			final double[] sigma,
			final double intensity )
	{
		final int numDimensions = image.numDimensions();
		final int[] size = new int[ numDimensions ];
		
		final long[] min = new long[ numDimensions ];
		final long[] max = new long[ numDimensions ];
		
		final double[] two_sq_sigma = new double[ numDimensions ];
		
		for ( int d = 0; d < numDimensions; ++d )
		{
			size[ d ] = Util.getSuggestedKernelDiameter( sigma[ d ] ) * 2;
			min[ d ] = (int)Math.round( location[ d ] ) - size[ d ]/2;
			max[ d ] = min[ d ] + size[ d ] - 1;
			two_sq_sigma[ d ] = 2 * sigma[ d ] * sigma[ d ];
		}

		final RandomAccessible< net.imglib2.type.numeric.real.FloatType > infinite = Views.extendZero( image );
		final RandomAccessibleInterval< net.imglib2.type.numeric.real.FloatType > interval = Views.interval( infinite, min, max );
		final IterableInterval< net.imglib2.type.numeric.real.FloatType > iterable = Views.iterable( interval );
		final Cursor< net.imglib2.type.numeric.real.FloatType > cursor = iterable.localizingCursor();
		
		while ( cursor.hasNext() )
		{
			cursor.fwd();
			
			double value = 1;
			
			for ( int d = 0; d < numDimensions; ++d )
			{
				final double x = location[ d ] - cursor.getIntPosition( d );
				value *= Math.exp( -(x * x) / two_sq_sigma[ d ] );
			}
			
			cursor.get().set( cursor.get().get() + ( (float)value * (float)intensity ) );
		}
	}

	/**
	 * Adds beads at random locations and returns a list of where they were ( the array is modified )
	 * 
	 * @param image - the image as float[] array
	 * @param dim - the dimensionality
	 * @param sigma - the sigma's of the beads to add
	 * @param numBeads - how many beads to add
	 * @param distanceToBorder - how far away from the image edge the beads can be
	 * @return - a list of where the beads are
	 */
	private static List< double[] > addBeads(
			final float[] image,
			final long[] dim,
			final double[] sigma,
			final int numBeads,
			final double distanceToBorder )
	{
		final int n = dim.length;
		
		final Img< net.imglib2.type.numeric.real.FloatType > img = ArrayImgs.floats( image, dim );

		// use a pseudo-random number so that the result is indentical every time
		final Random rnd = new Random( 435 );

		// base-level of gaussian-distributed noise
		for ( final net.imglib2.type.numeric.real.FloatType t : img )
			t.set( (float)Math.abs( rnd.nextGaussian() ) );

		final ArrayList< double[] > locations = new ArrayList<>();

		for ( int i = 0; i < numBeads; ++i )
		{
			final double loc[] = new double[ n ];

			for ( int d = 0; d < n; ++d )
				loc[ d ] = rnd.nextDouble() * (dim[ d ] - 2*distanceToBorder) + distanceToBorder;

			locations.add( loc );

			addGaussian( img, loc, sigma, 10 );
		}

		return locations;
	}

	private static double max( final float[] image )
	{
		double max = -1;

		for ( final float t : image )
			max = Math.max( max, t );

		return max;
	}

	private static void digitize( final float[] image, final int maxValue )
	{
		final double max = max( image );

		for ( int i = 0; i < image.length; ++i )
			image[ i ] = Math.round( ( image[ i ] / max ) * maxValue );
	}

    public static void main(String[] args)
    {
        final int width = 512;
        final int height = 512;
        final int depth = 100;
        final float calXY = 0.1625f;
        final float calZ = 0.2f;
        final int downsampleXY = 0; // 0 : a bit less then z-resolution, -1 : a bit more then z-resolution
        final int downsampleZ = 1;
        final float sigma = 1.8f;
        final float threshold = 0.05f;
        float[] image = new float[ width * height * depth ];
        int numPeaks = 300;

        // add 300 random beads (sigma=1.5,1.5,2.0, at least 5px to the border) and save where they were
        final List< double[] > beads = addBeads( image, new long[]{ width, height, depth }, new double[]{ 1.5, 1.5, 2 }, numPeaks, 5 );

        // scale the intensities between 0 ... 65535
        // (the threshold in the DoG is relative to the max intensity which is assumed to be 65535 here)
        digitize( image, 65535 );

        // display slightly saturated
        new ImageJ();
        ImageJFunctions.show( ArrayImgs.floats( image, new long[]{ width, height, depth } ) ).setDisplayRange( 0, 30000 );

        boolean useInteractiveDoGMethod = false;

        float[] interestPointsArray =
                compute(image, width, height, depth, calXY, calZ, downsampleXY, downsampleZ, sigma, threshold, useInteractiveDoGMethod);

        IOFunctions.println( "theExpectedPeakers = " + numPeaks );
        IOFunctions.println( "interestPointsArray.length / 3 = " + interestPointsArray.length / 3 );
        if( numPeaks == interestPointsArray.length / 3 )
        	System.out.println( "SUCCESS" );
        else
        	System.out.println( "FAIL" );
    }

    //
    // helper functions:
    //

    /**
     * Generate an legacy ImgLib image
     */

    private static Image<FloatType> createImage(int width, int height) {
        ImageFactory<FloatType> factory = new ImageFactory<FloatType>(new FloatType(), new ArrayContainerFactory());

        return factory.createImage(new int[]{width, height});
    }

    private static Image<FloatType> createPopulatedImage(int width, int height, float[] values) {
        Image<FloatType> image = createImage(width, height);

        LocalizableByDimCursor<FloatType> cursor = image.createLocalizableByDimCursor();

        int[] position = new int[2];

        int i = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                position[0] = x;
                position[1] = y;
                cursor.setPosition(position);
                cursor.getType().set(values[i++]);
            }
        }

        return image;
    }

    private static int downsampleFactor(final int downsampleXY, final int downsampleZ, final float calXY, final float calZ) {
        final double log2ratio = Math.log((calZ * downsampleZ) / calXY) / Math.log(2);

        final double exp2;

        if (downsampleXY == 0)
            exp2 = Math.pow(2, Math.floor(log2ratio));
        else
            exp2 = Math.pow(2, Math.ceil(log2ratio));

        return (int) Math.round(exp2);
    }

    //protected static RandomAccessibleInterval< net.imglib2.type.numeric.real.FloatType > downsample(
    protected static RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> downsample(
            Img<net.imglib2.type.numeric.real.FloatType> input,
            final float calXY,
            final float calZ,
            int downsampleXY,
            final int downsampleZ,
            final AffineTransform3D t) {
        // downsampleXY == 0 : a bit less then z-resolution
        // downsampleXY == -1 : a bit more then z-resolution
        if (downsampleXY < 1)
            downsampleXY = downsampleFactor(downsampleXY, downsampleZ, calXY, calZ);

        if (downsampleXY > 1)
            IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): Downsampling in XY " + downsampleXY + "x ...");

        if (downsampleZ > 1)
            IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): Downsampling in Z " + downsampleZ + "x ...");

        int dsx = downsampleXY;
        int dsy = downsampleXY;
        int dsz = downsampleZ;

        t.identity();

        final ImgFactory< net.imglib2.type.numeric.real.FloatType > f =
                ((Img<net.imglib2.type.numeric.real.FloatType>)input).factory();
        RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> output = input;

        t.set(downsampleXY, 0, 0);
        t.set(downsampleXY, 1, 1);
        t.set(downsampleZ, 2, 2);

        for (; dsx > 1; dsx /= 2)
            output = Downsample.simple2x(input, f, new boolean[]{true, false, false});

        for (; dsy > 1; dsy /= 2)
            output = Downsample.simple2x(input, f, new boolean[]{false, true, false});

        for (; dsz > 1; dsz /= 2)
            output = Downsample.simple2x(input, f, new boolean[]{false, false, true});

        return output;
    }

    /*
    //protected static RandomAccessibleInterval< net.imglib2.type.numeric.real.FloatType > downsample(
    protected static RandomAccessibleInterval<FloatType> downsample(
            Image<FloatType> input,
            final float calXY,
            final float calZ,
            int downsampleXY,
            final int downsampleZ,
            final AffineTransform3D t) {
        // downsampleXY == 0 : a bit less then z-resolution
        // downsampleXY == -1 : a bit more then z-resolution
        if (downsampleXY < 1)
            downsampleXY = downsampleFactor(downsampleXY, downsampleZ, calXY, calZ);

        if (downsampleXY > 1)
            IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): Downsampling in XY " + downsampleXY + "x ...");

        if (downsampleZ > 1)
            IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): Downsampling in Z " + downsampleZ + "x ...");

        int dsx = downsampleXY;
        int dsy = downsampleXY;
        int dsz = downsampleZ;

        t.identity();

        final ImageFactory< FloatType > f = input.getImageFactory();
        RandomAccessibleInterval<FloatType> output = input;

        t.set(downsampleXY, 0, 0);
        t.set(downsampleXY, 1, 1);
        t.set(downsampleZ, 2, 2);

        for (; dsx > 1; dsx /= 2)
            output = Downsample.simple2x(input, f, new boolean[]{true, false, false});

        for (; dsy > 1; dsy /= 2)
            output = Downsample.simple2x(input, f, new boolean[]{false, true, false});

        for (; dsz > 1; dsz /= 2)
            output = Downsample.simple2x(input, f, new boolean[]{false, false, true});

        return output;
    }
    */

    protected static <T extends RealType<T>> void preSmooth(final RandomAccessibleInterval<T> img,
                                                            final double additionalSigmaX,
                                                            final double additionalSigmaY,
                                                            final double additionalSigmaZ) {
        if (additionalSigmaX > 0.0 || additionalSigmaY > 0.0 || additionalSigmaZ > 0.0) {
            IOFunctions.println("presmoothing image with sigma=[" + additionalSigmaX + "," + additionalSigmaY + "," + additionalSigmaZ + "]");
            try {
                Gauss3.gauss(new double[]{additionalSigmaX, additionalSigmaY, additionalSigmaZ}, Views.extendMirrorSingle(img), img);
            } catch (IncompatibleTypeException e) {
                IOFunctions.println("presmoothing failed: " + e);
                e.printStackTrace();
            }
        }
    }
}
