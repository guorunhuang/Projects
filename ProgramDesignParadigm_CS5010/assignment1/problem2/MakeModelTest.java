package assignment1.problem2;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the MakeModel class.
 */
public class MakeModelTest {
  private MakeModel testMakeModel;
  private String expectedMake;
  private String expectedModel;

  @BeforeEach
  public void setUp() {
    this.expectedMake = "Benz";
    this.expectedModel = "Car";
    this.testMakeModel = new MakeModel(this.expectedMake, this.expectedModel);
  }

  @Test
  public void testGetMake() {
    assertEquals(this.expectedMake, this.testMakeModel.getMake());
  }

  @Test
  public void testGetModel() {
    assertEquals(this.expectedModel, this.testMakeModel.getModel());
  }
}