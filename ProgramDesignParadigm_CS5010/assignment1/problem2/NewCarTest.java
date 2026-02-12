package assignment1.problem2;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the NewCar class.
 */
public class NewCarTest {
  private NewCar testNewCar;
  private MakeModel testMakeModel;
  private String testId;
  private Integer testYear;
  private double testMsrp;
  private Integer testVehicleCount;

  @BeforeEach
  public void setUp() {
    this.testId = "NEW-123";
    this.testYear = 2024;
    this.testMsrp = 35000.0;
    this.testVehicleCount = 15;
    this.testMakeModel = new MakeModel("Honda", "Civic");
    this.testNewCar = new NewCar(this.testId, this.testYear, this.testMakeModel,
        this.testMsrp, this.testVehicleCount);
  }

  @Test
  public void testGetId() {
    assertEquals(this.testId, this.testNewCar.getId());
  }

  @Test
  public void testGetManufacturingYear() {
    assertEquals(this.testYear, this.testNewCar.getManufacturingYear());
  }

  @Test
  public void testGetMsrp() {
    assertEquals(this.testMsrp, this.testNewCar.getMsrp());
  }

  @Test
  public void testGetVehiclesWithin50Miles() {
    assertEquals(this.testVehicleCount, this.testNewCar.getVehiclesWithin50Miles());
  }
}