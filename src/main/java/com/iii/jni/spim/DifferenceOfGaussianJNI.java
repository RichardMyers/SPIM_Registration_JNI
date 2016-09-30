package com.iii.jni.spim;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

import ij.ImageJ;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyMirrorFactory;
import mpicbg.imglib.type.numeric.real.FloatType;
import mpicbg.imglib.wrapper.ImgLib2;
import mpicbg.spim.io.IOFunctions;
import mpicbg.spim.registration.bead.laplace.LaPlaceFunctions;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import spim.Threads;
import spim.fiji.spimdata.interestpoints.InterestPoint;
import spim.process.fusion.FusionHelper;
import spim.process.interestpointdetection.DifferenceOfGaussianNewPeakFinder;
import spim.process.interestpointdetection.Downsample;
import spim.process.interestpointdetection.Localization;
import spim.process.interestpointdetection.ProcessDOG;

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
        // scale the intensities between 0 ... 65535
        // (the threshold in the DoG is relative to the max intensity which is assumed to be 65535 here)
        // digitize( inImage, 65535 );

        final Img<net.imglib2.type.numeric.real.FloatType> image = ArrayImgs.floats( inImage, new long[]{ width, height, depth } );

        // display slightly saturated
        // new ImageJ();
        // ImageJFunctions.show( image ).setDisplayRange( 0, 30000 );

        float[] interestPointsArray = null;
        try {
            if (useInteractiveDoGMethod) {
                // use example from InteractiveDoG.java
                interestPointsArray = directDoG(image, calXY, calZ, downsampleXY, downsampleZ, sigma1, threshold);
            } else {
                // use example from DifferenceOfGaussian.java
                interestPointsArray = DifferenceOfGaussian(image, calXY, calZ, downsampleXY, downsampleZ, sigma1, threshold);
            }
        } catch (Exception e) {
            IOFunctions.println("DifferenceOfGaussianJNI.compute:: failed. " + e);
            e.printStackTrace();
        }

        return interestPointsArray;
    }

    public static float[] directDoG(final Img<net.imglib2.type.numeric.real.FloatType> image, final float calXY, final float calZ, final int downsampleXY, final int downsampleZ,
                                         final float intialSigma, final float threshold)
    {
        IOFunctions.println( "Using DifferenceOfGaussian directly");

        // down sample 'data' to create 'input'
        final AffineTransform3D affineTransform = new AffineTransform3D();
        final RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> input =
                downsample(image, calXY, calZ, downsampleXY, downsampleZ, affineTransform);

        // normalize the input so the values are comparable
        FusionHelper.normalizeImage( input, 0, 65535 );

        final Image<FloatType> img = ImgLib2.wrapFloatToImgLib1((Img<net.imglib2.type.numeric.real.FloatType>) input);

        final float k = LaPlaceFunctions.computeK( 4 );
		final float K_MIN1_INV = LaPlaceFunctions.computeKWeight(k);
		final int steps = 3;
		
		//
		// Compute the Sigmas for the gaussian convolution
		//
		final float[] sigmaStepsX = LaPlaceFunctions.computeSigma( steps, k, intialSigma );
		final float[] sigmaStepsDiffX = LaPlaceFunctions.computeSigmaDiff( sigmaStepsX, (float)0.5 );
		
		final float[] sigmaStepsY = LaPlaceFunctions.computeSigma( steps, k, intialSigma );
		final float[] sigmaStepsDiffY = LaPlaceFunctions.computeSigmaDiff( sigmaStepsY, (float)0.5 );
		
		final float[] sigmaStepsZ = LaPlaceFunctions.computeSigma( steps, k, intialSigma );
		final float[] sigmaStepsDiffZ = LaPlaceFunctions.computeSigmaDiff( sigmaStepsZ, (float)0.5 );
		
		final double[] sigma1 = new double[]{ sigmaStepsDiffX[0], sigmaStepsDiffY[0], sigmaStepsDiffZ[0] };
		final double[] sigma2 = new double[]{ sigmaStepsDiffX[1], sigmaStepsDiffY[1], sigmaStepsDiffZ[1] };
		
		final float minInitialPeakValue = threshold/10.0f;

		IOFunctions.println( "(" + new Date(System.currentTimeMillis()) + "): computing difference-of-gausian (sigma=0.5, " +
				"threshold=" + threshold + ", sigma1=" + Util.printCoordinates( sigma1 ) + ", sigma2=" + Util.printCoordinates( sigma2 ) + ")" );

		final DifferenceOfGaussianNewPeakFinder dog = new DifferenceOfGaussianNewPeakFinder( img, new OutOfBoundsStrategyMirrorFactory<FloatType>(), sigma1, sigma2, minInitialPeakValue, K_MIN1_INV );
		dog.setComputeConvolutionsParalell( false );
		dog.setNumThreads( Threads.numThreads() );
		dog.setKeepDoGImage( true );
		dog.process();

		final ArrayList< InterestPoint > finalPeaks = Localization.computeQuadraticLocalization( dog.getSimplePeaks(), dog.getDoGImage(), false, true, threshold, false );
		dog.getDoGImage().close();

        float[] interestPointsArray = new float[ finalPeaks.size() * 3];

        for (int i = 0; i < finalPeaks.size(); i++)
        {
            interestPointsArray[i * 3] = finalPeaks.get(i).getFloatPosition( 0);
            interestPointsArray[i * 3 + 1] = finalPeaks.get(i).getFloatPosition(1);
            interestPointsArray[i * 3 + 2] = finalPeaks.get(i).getFloatPosition(2);
        }

        correctForDownsampling( interestPointsArray, affineTransform );

        return interestPointsArray;
    }

    protected static void correctForDownsampling( final float[] ips, final AffineTransform3D t )
	{
		IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): Correcting coordinates for downsampling using AffineTransform: " + t );

		if ( ips == null || ips.length== 0 )
		{
			IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): WARNING: List is empty." );
			return;
		}

		final double[] tmp = new double[ 3 ];
		final double[] l = new double[ 3 ];

		for ( int i = 0; i < ips.length / 3; ++i )
		{
			l[ 0 ] = ips[ i * 3 ];
			l[ 1 ] = ips[ i * 3 + 1 ];
			l[ 2 ] = ips[ i * 3 + 2 ];

			t.apply( l, tmp );

			ips[ i * 3 ] = (float)tmp[ 0 ];
			ips[ i * 3 + 1 ] = (float)tmp[ 1 ];
			ips[ i * 3 + 2 ] = (float)tmp[ 2 ];
		}
	}

    public static float[] DifferenceOfGaussian(final Img<net.imglib2.type.numeric.real.FloatType> image,
                                               final float calXY, final float calZ,
                                               final int downsampleXY, final int downsampleZ,
                                               final float sigma1, final float threshold)
    {
        IOFunctions.println( "Using DifferenceOfGaussian method.");

        // down sample 'data' to create 'input'
        final AffineTransform3D affineTransform = new AffineTransform3D();
        final RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> input =
                downsample(image, calXY, calZ, downsampleXY, downsampleZ, affineTransform);

        // pre smooth data
        double additionalSigmaX = 0.0;
        double additionalSigmaY = 0.0;
        double additionalSigmaZ = 0.0;
        preSmooth(input, additionalSigmaX, additionalSigmaY, additionalSigmaZ);

        // wrap 'input' to create imglib1 'img'
        final Image<FloatType> img = ImgLib2.wrapFloatToImgLib1((Img<net.imglib2.type.numeric.real.FloatType>) input);

        ArrayList< InterestPoint >interestPoints = ProcessDOG.compute(
                null, // cuda
                null, // deviceList
                false, // accurateCUDA
                0, // percentGPUMem
                img,
                (Img<net.imglib2.type.numeric.real.FloatType>) input,
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

        correctForDownsampling( interestPointsArray, affineTransform );

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
	 * @param intensity - relative intensity of the beads over the noise
	 * @param distanceToBorder - how far away from the image edge the beads can be
	 * @return - a list of where the beads are
	 */
	private static List< double[] > addBeads(
			final float[] image,
			final long[] dim,
			final double[] sigma,
			final int numBeads,
			final double intensity,
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

			addGaussian( img, loc, sigma, intensity );
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

	final static public double squareDistance( final double[] p1, final double[] p2 )
	{
		double sum = 0.0;
		for ( int i = 0; i < p1.length; ++i )
		{
			final double d = p1[ i ] - p2[ i ];
			sum += d * d;
		}
		return sum;
	}

	/**
	 * How many of the detected points are within a certain distance of a known point
	 * 
	 * @param locations - the simulated locations
	 * @param interestPointsArray - the found locations
	 * @param tolerance - the tolerance distance (in px)
	 * @return how many points are correct given the tolerance
	 */
	private static int testCorrectLocations( final List< double[] > locations, final float[] interestPointsArray, final double tolerance )
	{
		final double[] l = new double[ 3 ];
		final double squareTolerance = tolerance * tolerance;

		int correctLocations = 0;

		// this is very inefficient and could use a tree structure instead of stupid search
		for ( int i = 0; i < interestPointsArray.length/3; ++i )
		{
			l[ 0 ] = interestPointsArray[ i * 3 ];
			l[ 1 ] = interestPointsArray[ i * 3 + 1 ];
			l[ 2 ] = interestPointsArray[ i * 3 + 2 ];

			boolean found = false;

			for ( int j = 0; j < locations.size() && !found; ++j )
				if ( squareDistance( locations.get( j ), l ) <= squareTolerance )
					found = true;

			if ( found )
				++correctLocations;
		}

		return correctLocations;
	}

    public static void main(String[] args)
    {
        final int width = 512;
        final int height = 512;
        final int depth = 100;
        final float calXY = 0.1625f;
        final float calZ = 0.4f; // 0.2 == no downsampling, 0.4 == downsampling 2x in XY
        final int downsampleXY = 0; // 0 : a bit less then z-resolution, -1 : a bit more then z-resolution
        final int downsampleZ = 1;
        final float sigma = 1.5f;
        final float threshold = 0.05f;
        float[] image = new float[ width * height * depth ];
        int numPeaks = 200;

        // add 300 random beads (sigma=1.5,1.5,2.0, at least 5px to the border) and save where they were
        final List< double[] > beads = addBeads( image, new long[]{ width, height, depth }, new double[]{ 1.5, 1.5, 2 }, numPeaks, 10, 5 );

        // scale the intensities between 0 ... 65535
        // (the threshold in the DoG is relative to the max intensity which is assumed to be 65535 here)
        digitize( image, 65535 );

        // display slightly saturated
        new ImageJ();
        ImageJFunctions.show( ArrayImgs.floats( image, new long[]{ width, height, depth } ) ).setDisplayRange( 0, 30000 );

        boolean useInteractiveDoGMethod = true;

        float[] interestPointsArray =
                compute(image, width, height, depth, calXY, calZ, downsampleXY, downsampleZ, sigma, threshold, useInteractiveDoGMethod);

        IOFunctions.println( "theExpectedPeakers = " + numPeaks );
        IOFunctions.println( "interestPointsArray.length / 3 = " + interestPointsArray.length / 3 );
        IOFunctions.println( testCorrectLocations( beads, interestPointsArray, 0.5 ) + " (out of " + interestPointsArray.length / 3 + ") detections are correctly found with an error of 0.5 px" );

        if( numPeaks == interestPointsArray.length / 3 )
        	System.out.println( "SUCCESS" );
        else
        	System.out.println( "FAIL" );
    }

    //
    // helper functions:
    //

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
            output = Downsample.simple2x(output, f, new boolean[]{false, true, false});

        for (; dsz > 1; dsz /= 2)
            output = Downsample.simple2x(output, f, new boolean[]{false, false, true});

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
