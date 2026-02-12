package assignment1.problem2;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class UsedCarTest {
  private UsedCar testCar;
  private MakeModel mm;

  @BeforeEach
  public void setUp() {
    mm = new MakeModel("Benz", "Car");
    testCar = new UsedCar("CAR123", 2020, mm, 25000.0, 30000, 1, 0);
  }

  @Test
  public void testGetId() {
    assertEquals("CAR123", testCar.getId());
  }

  @Test
  public void testGetMileage() {
    assertEquals(30000, testCar.getMileage());
  }

  @Test
  public void testGetMake() {
    assertEquals("Benz", testCar.getMakeModel().getMake());
  }
}