package com.iii.jni.spim;

import java.util.*;
import java.awt.Rectangle;

import ij.ImageJ;
import mpicbg.imglib.algorithm.scalespace.SubpixelLocalization;
import mpicbg.imglib.container.array.ArrayContainerFactory;
import mpicbg.imglib.cursor.LocalizableCursor;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.image.ImageFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyMirrorFactory;
import mpicbg.imglib.type.numeric.real.FloatType;
import mpicbg.imglib.wrapper.ImgLib2;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.io.IOFunctions;
import mpicbg.spim.registration.bead.laplace.LaPlaceFunctions;
import mpicbg.spim.registration.detection.DetectionSegmentation;
import mpicbg.spim.registration.ViewStructure;
import mpicbg.imglib.algorithm.scalespace.DifferenceOfGaussianPeak;
import mpicbg.imglib.algorithm.scalespace.DifferenceOfGaussianReal1;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyValueFactory;
import net.imglib2.*;
import net.imglib2.RandomAccess;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.img.imageplus.FloatImagePlus;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import spim.Threads;
import spim.fiji.spimdata.interestpoints.InterestPoint;
import spim.fiji.spimdata.SpimData2;
import spim.fiji.spimdata.XmlIoSpimData2;
import spim.fiji.spimdata.SpimData2;
import spim.process.fusion.FusionHelper;
import spim.process.fusion.boundingbox.BoundingBoxGUI;
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
								  final boolean useInteractiveDoGMethod,
								  final int currentPlane) {
		// display slightly saturated
		// new ImageJ();
		// ImageJFunctions.show( image ).setDisplayRange( 0, 30000 );

		// display slightly saturated
		//new ImageJ();
		//ImageJFunctions.show( ArrayImgs.floats( inImage, new long[]{ width, height, depth } ) ).setDisplayRange( 0, 30000 );

		IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): DifferenceOfGaussianJNI.compute (inImage, width=" + width + ", height =" + height +
				", depth=" + depth + ", calXY=" + calXY + ", calZ=" + calZ + ", downsampleXY=" + downsampleXY + ", downsampleZ=" + downsampleZ + ", sigma1=" + sigma1 + ", threshold=" + threshold +
				", useInteractiveDoGMethod=" + useInteractiveDoGMethod + ", currentPlane=" + currentPlane + ")");

		float sigma1_tmp = sigma1;
		if (sigma1 <= 0.0f || sigma1 > 256.0f) {
			sigma1_tmp = 1.8f;
			IOFunctions.println("ERROR: sigma1 <= 0.0 || sigma1 > 256.0");
		}

		float threshold_tmp = threshold;
		if (threshold <= 0.0f || threshold > 10.0f) {
			threshold_tmp = 1.8f;
			IOFunctions.println("ERROR: threshold <= 0.0 || threshold > 10.0f");
		}

		float[] interestPointsArray = null;
		try {
			if (useInteractiveDoGMethod) {
				// use example from InteractiveDoG.java
				final Img<net.imglib2.type.numeric.real.FloatType> image = ArrayImgs.floats(inImage, new long[]{width, height, depth});

				FusionHelper.normalizeImage(image);

				interestPointsArray = directDoG(image, calXY, calZ, downsampleXY, downsampleZ, sigma1_tmp, threshold_tmp, currentPlane);

				// use example from DetectSegmentation.java -- creates too many points
				// interestPointsArray = DetectionSegmentation(image, calXY, calZ, downsampleXY, downsampleZ, sigma1, threshold, currentPlane);
			} else {
				// scale the intensities between 0 ... 65535
				// (the threshold in the DoG is relative to the max intensity which is assumed to be 65535 here)
				digitize(inImage, 65535);

				final Img<net.imglib2.type.numeric.real.FloatType> image = ArrayImgs.floats(inImage, new long[]{width, height, depth});
				// use example from DifferenceOfGaussian.java
				interestPointsArray = DifferenceOfGaussian(image, calXY, calZ, downsampleXY, downsampleZ, sigma1, threshold, currentPlane);
			}
		} catch (Exception e) {
			IOFunctions.println("DifferenceOfGaussianJNI.compute:: failed. " + e);
			e.printStackTrace();
		}

		return interestPointsArray;
	}

	// use example from DetectSegmentation.java
	public static float[] DetectionSegmentation(final Img<net.imglib2.type.numeric.real.FloatType> image,
												final float calXY, final float calZ,
												final int downsampleXY, final int downsampleZ,
												final float sigma1, final float threshold, final int currentPlane) {
		IOFunctions.println("Using DetectSegmentation method.");

		float imageSigma = 0.5f;
		int sensitivity = 4;
		float k = (float) DetectionSegmentation.computeK(sensitivity);
		// K_MIN1_INV = DetectionSegmentation.computeKWeight( k );
		// float sigmaDiff = DetectionSegmentation.computeSigmaDiff( sigma, imageSigma );
		float[] sigma = DetectionSegmentation.computeSigma(k, sigma1);
		float sigma2 = sigma[1];
		final Image<FloatType> img = ImgLib2.wrapFloatToImgLib1((Img<net.imglib2.type.numeric.real.FloatType>) image);

		final ArrayList<DifferenceOfGaussianPeak<FloatType>> peaks =
				DetectionSegmentation.extractBeadsLaPlaceImgLib(
						img,
						new OutOfBoundsStrategyMirrorFactory<FloatType>(),
						imageSigma,
						sigma1,
						sigma2,
						threshold,
						threshold / 4,
						true,
						false,
						ViewStructure.DEBUG_MAIN);

		float[] interestPointsArray = new float[peaks.size() * 3];

		for (int i = 0; i < peaks.size(); i++) {
			interestPointsArray[i * 3] = peaks.get(i).get(0);
			interestPointsArray[i * 3 + 1] = peaks.get(i).get(1);
			interestPointsArray[i * 3 + 2] = peaks.get(i).get(2);
		}

		return interestPointsArray;
	}

	protected static Image<FloatType> extractImage(final Img<net.imglib2.type.numeric.real.FloatType> source, final Rectangle rectangle, final int extraSize, final int inPlane) {
		final Image<FloatType> img = new ImageFactory<FloatType>(new FloatType(), new ArrayContainerFactory()).createImage(new int[]{rectangle.width + extraSize, rectangle.height + extraSize});

		final int offsetX = rectangle.x - extraSize / 2;
		final int offsetY = rectangle.y - extraSize / 2;

		final int[] location = new int[source.numDimensions()];

		if (location.length > 2)
			location[2] = inPlane;

		final LocalizableCursor<FloatType> cursor = img.createLocalizableCursor();
		final RandomAccess<net.imglib2.type.numeric.real.FloatType> positionable;

		if (offsetX >= 0 && offsetY >= 0 &&
				offsetX + img.getDimension(0) < source.dimension(0) &&
				offsetY + img.getDimension(1) < source.dimension(1)) {
			// it is completely inside so we need no outofbounds for copying
			positionable = source.randomAccess();
		} else {
			positionable = Views.extendMirrorSingle(source).randomAccess();
		}

		while (cursor.hasNext()) {
			cursor.fwd();
			cursor.getPosition(location);

			location[0] += offsetX;
			location[1] += offsetY;

			positionable.setPosition(location);

			cursor.getType().set(positionable.get().get());
		}

		return img;
	}

	public static float[] directDoG(final Img<net.imglib2.type.numeric.real.FloatType> image, final float calXY, final float calZ, final int downsampleXY, final int downsampleZ,
									final float initialSigma, final float threshold, final int currentPlane) {
		IOFunctions.println("Using directDoG method");

		// down sample 'data' to create 'input'
		//final AffineTransform3D affineTransform = new AffineTransform3D();
		//final RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> input =
		//		downsample(image, calXY, calZ, downsampleXY, downsampleZ, affineTransform);

		//
		// extract middle plane
		//

		final int extraSize = 40;
		final Rectangle rect = new Rectangle(0, 0, (int) image.dimension(0), (int) image.dimension(1));
		final Image<FloatType> img = extractImage(image, rect, extraSize, currentPlane);

		/*
		final Image<FloatType> img = new ImageFactory<FloatType>( new FloatType(), new ArrayContainerFactory() ).createImage( new int[]{ (int) image.dimension(0), (int) image.dimension(1) } );

		final int[] location = new int[ image.numDimensions() ];

		if ( location.length > 2 )
			location[ 2 ] = (int)(currentPlane); // middle plane

		final LocalizableCursor<FloatType> cursor = img.createLocalizableCursor();
		final RandomAccess<net.imglib2.type.numeric.real.FloatType> positionable;

		// it is completely inside so we need no outofbounds for copying
		positionable = image.randomAccess();

		while ( cursor.hasNext() )
		{
			cursor.fwd();
			cursor.getPosition( location );

			positionable.setPosition( location );

			cursor.getType().set( positionable.get().get() );
		}
		*/

		//
		// Compute the Sigmas for the gaussian convolution
		//

		float imageSigma = 0.5f;
		int sensitivity = 4;
		float k = (float) DetectionSegmentation.computeK(sensitivity);
		float K_MIN1_INV = DetectionSegmentation.computeKWeight(k);
		float[] sigma = DetectionSegmentation.computeSigma(k, initialSigma);
		float[] sigmaDiff = DetectionSegmentation.computeSigmaDiff(sigma, imageSigma);

		//final float[] sigmaStepsX = LaPlaceFunctions.computeSigma( steps, k, initialSigma );
		//final float[] sigmaStepsDiffX = LaPlaceFunctions.computeSigmaDiff( sigmaStepsX, (float)imageSigma );

		IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): computing difference-of-gausian (sigma=0.5, " +
				"threshold=" + threshold + ", initial sigma = " + initialSigma + ", sigma1=" + sigmaDiff[0] + ", sigma2=" + sigmaDiff[1] + ")");

		float thresholdMin = 0.0001f;
		final DifferenceOfGaussianReal1<FloatType> dog = new DifferenceOfGaussianReal1<>(img, new OutOfBoundsStrategyValueFactory<FloatType>(), sigmaDiff[0], sigmaDiff[1], thresholdMin / 4, K_MIN1_INV);
		dog.setKeepDoGImage(true);
		dog.process();

		final SubpixelLocalization<FloatType> subpixel = new SubpixelLocalization<>(dog.getDoGImage(), dog.getPeaks());
		subpixel.process();

		ArrayList<DifferenceOfGaussianPeak<FloatType>> peaks = dog.getPeaks();

		ArrayList<DifferenceOfGaussianPeak<FloatType>> finalPeaks = new ArrayList<DifferenceOfGaussianPeak<FloatType>>();
		for (final DifferenceOfGaussianPeak<FloatType> peak : peaks) {
			final float x = peak.getPosition(0);
			final float y = peak.getPosition(1);

			if (peak.isMax() && Math.abs(peak.getValue().get()) > threshold &&
					x >= extraSize / 2 && y >= extraSize / 2 &&
					x < rect.width + extraSize / 2 && y < rect.height + extraSize / 2) {
				finalPeaks.add(peak);
			}
		}

		float[] interestPointsArray = new float[finalPeaks.size() * 3];
		for (int i = 0; i < finalPeaks.size(); i++) {
			interestPointsArray[i * 3] = finalPeaks.get(i).getPosition(0) + rect.x - extraSize / 2;
			interestPointsArray[i * 3 + 1] = finalPeaks.get(i).getPosition(1) + rect.y - extraSize / 2;
			;
			interestPointsArray[i * 3 + 2] = currentPlane;
		}

		// correctForDownsampling( interestPointsArray, affineTransform );

		return interestPointsArray;
	}

	protected static void correctForDownsampling(final float[] ips, final AffineTransform3D t) {
		IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): Correcting coordinates for downsampling using AffineTransform: " + t);

		if (ips == null || ips.length == 0) {
			IOFunctions.println("(" + new Date(System.currentTimeMillis()) + "): WARNING: List is empty.");
			return;
		}

		final double[] tmp = new double[3];
		final double[] l = new double[3];

		for (int i = 0; i < ips.length / 3; ++i) {
			l[0] = ips[i * 3];
			l[1] = ips[i * 3 + 1];
			l[2] = ips[i * 3 + 2];

			t.apply(l, tmp);

			ips[i * 3] = (float) tmp[0];
			ips[i * 3 + 1] = (float) tmp[1];
			ips[i * 3 + 2] = (float) tmp[2];
		}
	}

	public static float[] DifferenceOfGaussian(final Img<net.imglib2.type.numeric.real.FloatType> image,
											   final float calXY, final float calZ,
											   final int downsampleXY, final int downsampleZ,
											   final float sigma1, final float threshold, final int currentPlane) {
		IOFunctions.println("Using DifferenceOfGaussian method.");

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

		ArrayList<InterestPoint> interestPoints = ProcessDOG.compute(
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

		correctForDownsampling(interestPointsArray, affineTransform);

		return interestPointsArray;
	}

	final public static void addGaussian(
			final Img<net.imglib2.type.numeric.real.FloatType> image,
			final double[] location,
			final double[] sigma,
			final double intensity) {
		final int numDimensions = image.numDimensions();
		final int[] size = new int[numDimensions];

		final long[] min = new long[numDimensions];
		final long[] max = new long[numDimensions];

		final double[] two_sq_sigma = new double[numDimensions];

		for (int d = 0; d < numDimensions; ++d) {
			size[d] = Util.getSuggestedKernelDiameter(sigma[d]) * 2;
			min[d] = (int) Math.round(location[d]) - size[d] / 2;
			max[d] = min[d] + size[d] - 1;
			two_sq_sigma[d] = 2 * sigma[d] * sigma[d];
		}

		final RandomAccessible<net.imglib2.type.numeric.real.FloatType> infinite = Views.extendZero(image);
		final RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> interval = Views.interval(infinite, min, max);
		final IterableInterval<net.imglib2.type.numeric.real.FloatType> iterable = Views.iterable(interval);
		final Cursor<net.imglib2.type.numeric.real.FloatType> cursor = iterable.localizingCursor();

		while (cursor.hasNext()) {
			cursor.fwd();

			double value = 1;

			for (int d = 0; d < numDimensions; ++d) {
				final double x = location[d] - cursor.getIntPosition(d);
				value *= Math.exp(-(x * x) / two_sq_sigma[d]);
			}

			cursor.get().set(cursor.get().get() + ((float) value * (float) intensity));
		}
	}

	/**
	 * Adds beads at random locations and returns a list of where they were ( the array is modified )
	 *
	 * @param image            - the image as float[] array
	 * @param dim              - the dimensionality
	 * @param sigma            - the sigma's of the beads to add
	 * @param numBeads         - how many beads to add
	 * @param intensity        - relative intensity of the beads over the noise
	 * @param distanceToBorder - how far away from the image edge the beads can be
	 * @return - a list of where the beads are
	 */
	private static List<double[]> addBeads(
			final float[] image,
			final long[] dim,
			final double[] sigma,
			final int numBeads,
			final double intensity,
			final double distanceToBorder) {
		final int n = dim.length;

		final Img<net.imglib2.type.numeric.real.FloatType> img = ArrayImgs.floats(image, dim);

		// use a pseudo-random number so that the result is indentical every time
		final Random rnd = new Random(435);

		// base-level of gaussian-distributed noise
		for (final net.imglib2.type.numeric.real.FloatType t : img)
			t.set((float) Math.abs(rnd.nextGaussian()));

		final ArrayList<double[]> locations = new ArrayList<>();

		for (int i = 0; i < numBeads; ++i) {
			final double loc[] = new double[n];

			for (int d = 0; d < n; ++d)
				loc[d] = rnd.nextDouble() * (dim[d] - 2 * distanceToBorder) + distanceToBorder;

			locations.add(loc);

			addGaussian(img, loc, sigma, intensity);
		}

		return locations;
	}

	private static double max(final float[] image) {
		double max = -1;

		for (final float t : image)
			max = Math.max(max, t);

		return max;
	}

	private static void digitize(final float[] image, final int maxValue) {
		final double max = max(image);

		for (int i = 0; i < image.length; ++i)
			image[i] = Math.round((image[i] / max) * maxValue);
	}

	final static public double squareDistance(final double[] p1, final double[] p2) {
		double sum = 0.0;
		for (int i = 0; i < p1.length; ++i) {
			final double d = p1[i] - p2[i];
			sum += d * d;
		}
		return sum;
	}

	/**
	 * How many of the detected points are within a certain distance of a known point
	 *
	 * @param locations           - the simulated locations
	 * @param interestPointsArray - the found locations
	 * @param tolerance           - the tolerance distance (in px)
	 * @return how many points are correct given the tolerance
	 */
	private static int testCorrectLocations(final List<double[]> locations, final float[] interestPointsArray, final double tolerance) {
		final double[] l = new double[3];
		final double squareTolerance = tolerance * tolerance;

		int correctLocations = 0;

		// this is very inefficient and could use a tree structure instead of stupid search
		for (int i = 0; i < interestPointsArray.length / 3; ++i) {
			l[0] = interestPointsArray[i * 3];
			l[1] = interestPointsArray[i * 3 + 1];
			l[2] = interestPointsArray[i * 3 + 2];

			boolean found = false;

			for (int j = 0; j < locations.size() && !found; ++j)
				if (squareDistance(locations.get(j), l) <= squareTolerance)
					found = true;

			if (found)
				++correctLocations;
		}

		return correctLocations;
	}

	public static void main(String[] args) {
		final int width = 512;
		final int height = 512;
		final int depth = 100;
		final float calXY = 0.1625f;
		final float calZ = 0.4f; // 0.2 == no downsampling, 0.4 == downsampling 2x in XY, 0.8 == downsampling 4x in XY
		final int downsampleXY = 0; // 0 : a bit less then z-resolution, -1 : a bit more then z-resolution
		final int downsampleZ = 1;
		final float sigma = 1.5f;
		final float threshold = 0.05f;
		float[] image = new float[width * height * depth];
		int numPeaks = 200;

		// add 300 random beads (sigma=1.5,1.5,2.0, at least 5px to the border) and save where they were
		final List<double[]> beads = addBeads(image, new long[]{width, height, depth}, new double[]{1.5, 1.5, 2}, numPeaks, 10, 5);

		IOFunctions.println("width = " + width + " height = " + height + " depth = " + depth + " calXY = " + calXY + " calZ = " + calZ);
		IOFunctions.println("downsampleXY = " + downsampleXY + " downsampleZ = " + downsampleZ + " sigma = " + sigma + " threshold = " + threshold);

		boolean useInteractiveDoGMethod;
		float[] interestPointsArray;

		{
			useInteractiveDoGMethod = true;

			interestPointsArray =
					compute(image, width, height, depth, calXY, calZ, downsampleXY, downsampleZ, sigma, threshold, useInteractiveDoGMethod, depth / 2);

			IOFunctions.println("theExpectedPeakers = " + numPeaks);
			IOFunctions.println("interestPointsArray.length / 3 = " + interestPointsArray.length / 3);
			IOFunctions.println(testCorrectLocations(beads, interestPointsArray, 0.5) + " (out of " + interestPointsArray.length / 3 + ") detections are correctly found with an error of 0.5 px");

			if (numPeaks == interestPointsArray.length / 3) {
				System.out.println("SUCCESS");
				IOFunctions.println("SUCCESS");
			} else {
				System.out.println("FAIL");
				IOFunctions.println("FAIL");
			}
		}

		{
			useInteractiveDoGMethod = false;

			interestPointsArray =
					compute(image, width, height, depth, calXY, calZ, downsampleXY, downsampleZ, sigma, threshold, useInteractiveDoGMethod, depth / 2);

			IOFunctions.println("theExpectedPeakers = " + numPeaks);
			IOFunctions.println("interestPointsArray.length / 3 = " + interestPointsArray.length / 3);
			IOFunctions.println(testCorrectLocations(beads, interestPointsArray, 0.5) + " (out of " + interestPointsArray.length / 3 + ") detections are correctly found with an error of 0.5 px");

			if (numPeaks == interestPointsArray.length / 3) {
				System.out.println("SUCCESS");
				IOFunctions.println("SUCCESS");
			} else {
				System.out.println("FAIL");
				IOFunctions.println("FAIL");
			}
		}

		{
			int[] rangeMin = new int[ 3 ];
			int[] rangeMax = new int[ 3 ];
			if (GetDefaultBoundingBox("G:\\3i\\diSPIM\\Warrick\\boundingbox.xml", 0, rangeMin, rangeMax)) {
				IOFunctions.println("rangeMin = " + rangeMin[0] + ", " + rangeMin[1] + ", " + rangeMin[2]);
				IOFunctions.println("rangeMax = " + rangeMax[0] + ", " + rangeMax[1] + ", " + rangeMax[2]);
				if (rangeMin[0] == -138 && rangeMin[1] == -129 && rangeMin[2] == -155 &&
					rangeMax[0] == 658 && rangeMax[1] == 719 && rangeMax[2] == 215) {
					System.out.println("SUCCESS");
				}
			}
			else {
				System.out.println("FAIL");
			}
		}
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

		final ImgFactory<net.imglib2.type.numeric.real.FloatType> f =
				((Img<net.imglib2.type.numeric.real.FloatType>) input).factory();
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

		protected static boolean GetDefaultBoundingBox(final String xmlFilename, final int setupId, int[] rangeMin, int[] rangeMax ) {
			boolean changedSpimDataObject = false;
			// first time point, selected setupId
			List<ViewId> viewIdsToProcess = Arrays.asList(new ViewId(0, setupId), new ViewId(0, setupId+1));

			try {
				// ask for everything
				final XmlIoSpimData2 io = new XmlIoSpimData2("");
				final SpimData2 spimData = io.load(xmlFilename);
				double[] minBB = new double[ rangeMin.length ];
				double[] maxBB =  new double[ rangeMin.length ];
				int[] minBound = null;
				int[] maxBound = null;

				changedSpimDataObject = BoundingBoxGUI.computeMaxBoundingBoxDimensions( spimData, viewIdsToProcess, minBB, maxBB );

				for ( int d = 0; d < minBB.length; ++d )
				{
					rangeMin[ d ] = Math.round( (float)Math.floor( minBB[ d ] ) );
					rangeMax[ d ] = Math.round( (float)Math.floor( maxBB[ d ] ) );

					if ( rangeMin[ d ] < 0 )
						--rangeMin[ d ];

					if ( rangeMax[ d ] > 0 )
						++rangeMax[ d ];

					// first time called on this object
					if ( minBound == null || maxBound == null )
					{
						minBound = new int[ rangeMin.length ];
						maxBound = new int[ rangeMin.length ];
					}

					if ( minBound[ d ] == 0 && maxBound[ d ] == 0 )
					{
						// not preselected
						//if ( BoundingBoxGUI.defaultMin[ d ] == 0 && BoundingBoxGUI.defaultMax[ d ] == 0 )
						{
							minBound[ d ] = rangeMin[ d ];
							maxBound[ d ] = rangeMax[ d ];
						}
					}

					if ( minBound[ d ] > maxBound[ d ] )
						minBound[ d ] = maxBound[ d ];

					if ( minBound[ d ] < rangeMin[ d ] )
						rangeMin[ d ] = minBound[ d ];

					if ( maxBound[ d ] > rangeMax[ d ] )
						rangeMax[ d ] = maxBound[ d ];

					// test if the values are valid
					// test if the values are valid
					//if ( min[ d ] < rangeMin[ d ] )
					//	min[ d ] = rangeMin[ d ];

					//if ( max[ d ] > rangeMax[ d ] )
					//	max[ d ] = rangeMax[ d ];

				}
			}
			catch (mpicbg.spim.data.SpimDataException e){
				return false;
			}
			return true;
	}
}