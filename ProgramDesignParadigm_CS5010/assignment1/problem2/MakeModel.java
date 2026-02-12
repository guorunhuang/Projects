package assignment1.problem2;

/**
 * Represents the make and model of a vehicle.
 */
public class MakeModel {
  private String make;
  private String model;

  /**
   * Constructor for MakeModel.
   * @param make the vehicle make
   * @param model the vehicle model
   */
  public MakeModel(String make, String model) {
    this.make = make;
    this.model = model;
  }

  /** @return the vehicle make */
  public String getMake() { return this.make; }

  /** @return the vehicle model */
  public String getModel() { return this.model; }
}