package com.iii.jni.spim;

import com.sun.tools.javac.util.Assert;
import mpicbg.imglib.container.array.ArrayContainerFactory;
import mpicbg.imglib.cursor.LocalizableByDimCursor;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.image.ImageFactory;
import mpicbg.imglib.type.numeric.real.FloatType;
import mpicbg.imglib.wrapper.ImgLib2;
import mpicbg.spim.io.IOFunctions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;
import spim.fiji.spimdata.interestpoints.InterestPoint;
import spim.process.interestpointdetection.Downsample;
import spim.process.interestpointdetection.ProcessDOG;

import java.util.*;

/**
 * Created by Richard on 8/16/2016.
 */
public class DifferenceOfGaussianJNI {

    // call this method from JNI, returns array with {x,y} sub-pixel points of interest
    public static float[] compute(float[] inImage, final int width, final int height,
                                  final float calXY, final float calZ,
                                  final int downsampleXY, final int downsampleZ,
                                  final float sigma, final float threshold)
    {
        ArrayList<InterestPoint> interestPoints;

        try {
            IOFunctions.println("HERE1");

            // create 'imageArray' which wraps 'inImage' float array
            net.imglib2.img.basictypeaccess.array.FloatArray imageArray = new FloatArray(inImage);

            IOFunctions.println("HERE2");

            // create 'data' ArrayImg from 'imageArray'
            ArrayImg<net.imglib2.type.numeric.real.FloatType, net.imglib2.img.basictypeaccess.array.FloatArray> data;
            data = new ArrayImg<>(imageArray, (new long[]{width, height}), new net.imglib2.util.Fraction());

            // down sample 'data' to create 'input'
            final AffineTransform3D affineTransform = new AffineTransform3D();
            final RandomAccessibleInterval<net.imglib2.type.numeric.real.FloatType> input =
                    downsample(data, calXY, calZ, downsampleXY, downsampleZ, affineTransform);
            IOFunctions.println("HERE3");

            // pre smooth data
            double additionalSigmaX = 0.0;
            double additionalSigmaY = 0.0;
            double additionalSigmaZ = 0.0;
            preSmooth(input, additionalSigmaX, additionalSigmaY, additionalSigmaZ);

            // wrap 'input' to create imglib1 'img'
            final Image<FloatType> img = ImgLib2.wrapFloatToImgLib1((Img<net.imglib2.type.numeric.real.FloatType>) input);

            IOFunctions.println("HERE4");

            interestPoints = ProcessDOG.compute(
                    null, // cuda
                    null, // deviceList
                    false, // accurateCUDA
                    0, // percentGPUMem
                    img,
                    (Img<net.imglib2.type.numeric.real.FloatType>) input,
                    sigma,
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

            float[] interestPointsArray = new float[interestPoints.size()*2];
            for (int i = 0; i < interestPoints.size(); i++) {
                interestPointsArray[i*2] = interestPoints.get(i).getFloatPosition(0);
                interestPointsArray[i*2+1] = interestPoints.get(i).getFloatPosition(0);
            }

            IOFunctions.println("HERE5");
            return interestPointsArray;
        }
        catch( Exception e )
        {
            IOFunctions.println( "DifferenceOfGaussianJNI:: failed to compute. " + e );
            e.printStackTrace();
        }

        IOFunctions.println("HERE6");

        return null;
    }

    public static void main(String[] args) {
        final int width = 512;
        final int height = 512;
        final float calXY = 0.1625f;
        final float calZ = 0.2f;
        final int downsampleXY = 0; // 0 : a bit less then z-resolution, -1 : a bit more then z-resolution
        final int downsampleZ = 1;
        final float sigma = 1.8f;
        final float threshold = 0.008f;
        float[] image = new float[width * height];

        Random rand = new Random();
        for (int i = 0; i < width * height; i++) {
            // noise with peak every 10x10 pixels
            int row = i / width + 32;
            int col = i - row * width + 32;
            if (row % 64 == 0 && col % 64 == 0 ) {
                image[i] = 2000f  + (rand.nextFloat() * 100f);
            }
            else {
                image[i] = 0f;
            }
        }

        float[] interestPointsArray =
                compute(image, width, height, calXY, calZ, downsampleXY, downsampleZ, sigma, threshold);

        int theExpectedPeaks = (width/64 * height/64);
        IOFunctions.println( "theExpectedPeakers = " + theExpectedPeaks );
        IOFunctions.println( "interestPointsArray.length / 2 = " + interestPointsArray.length / 2 );
        Assert.check(theExpectedPeaks == interestPointsArray.length / 2);
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

    /**
     * Generate an ImageLib2 image
     */
    private static <T extends RealType<T> & NativeType<T>> Img<T> makeImage(final T type, float[] values, final long[] dims) {
        final ImgFactory<T> factory = new ArrayImgFactory<T>();
        final Img<T> result = factory.create(dims, type);
        final net.imglib2.Cursor<T> cursor = result.cursor();
        final long[] pos = new long[cursor.numDimensions()];
        while (cursor.hasNext()) {
            cursor.fwd();
            cursor.localize(pos);
            final long index = pos[1] * dims[0] + pos[0];
            final float value = values[(int) index];
            cursor.get().setReal(value);
        }
        return result;
    }

    private static Img<net.imglib2.type.numeric.real.FloatType> makeTestImage2D(long width, long height, float[] values) {
        return makeImage(new net.imglib2.type.numeric.real.FloatType(), values, new long[]{width, height});
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
            ArrayImg<net.imglib2.type.numeric.real.FloatType, ?> input,
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

        final ImgFactory<net.imglib2.type.numeric.real.FloatType> f = (input).factory();
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
