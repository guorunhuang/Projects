package assignment1.problem2;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the Boat class.
 */
public class BoatTest {
  private Boat testBoat;
  private MakeModel testMakeModel;
  private float testLength;
  private Integer testPassengers;
  private PropulsionType testPropulsion;

  @BeforeEach
  public void setUp() {
    this.testMakeModel = new MakeModel("Edge", "Boat");
    this.testLength = 17.5f;
    this.testPassengers = 6;
    this.testPropulsion = PropulsionType.OUTBOARD_ENGINE;
    this.testBoat = new Boat("BOAT-99", 2022, this.testMakeModel, 45000.0,
        this.testLength, this.testPassengers, this.testPropulsion);
  }

  @Test
  public void testGetLength() {
    assertEquals(this.testLength, this.testBoat.getLength());
  }

  @Test
  public void testGetPassengers() {
    assertEquals(this.testPassengers, this.testBoat.getPassengers());
  }

  @Test
  public void testGetPropulsionType() {
    assertEquals(this.testPropulsion, this.testBoat.getPropulsionType());
  }

  @Test
  public void testGetMakeModel() {
    assertEquals(this.testMakeModel, this.testBoat.getMakeModel());
  }
}