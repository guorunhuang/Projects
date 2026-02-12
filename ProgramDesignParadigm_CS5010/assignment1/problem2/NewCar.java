package assignment1.problem2;

/**
 * Represents a new car in inventory.
 */
public class NewCar extends Vehicle {
  private Integer vehiclesWithin50Miles;

  public NewCar(String id, Integer year, MakeModel mm, double msrp, Integer count) {
    super(id, year, mm, msrp);
    this.vehiclesWithin50Miles = count;
  }

  public Integer getVehiclesWithin50Miles() {
    return this.vehiclesWithin50Miles;
  }
}